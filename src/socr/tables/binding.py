"""Binding oracle — geometric cell-level binding between a born-digital page's
native word layer and a model-authored markdown grid.

Pure module. No I/O, no ``fitz`` import, no model calls. Input is exactly what
``page.get_text("words")`` returns (a list of ``(x0, y0, x1, y1, text, block,
line, word)`` tuples) plus the candidate's authored markdown text.

## Why this exists

Every multiset-based check in this codebase (the old ``_value_guard`` in
``native_verifier.py``) is blind to the failure mode that actually ships
wrong: a **flattened** table has an identical numeric multiset to a correctly
bound one — the values are all present, just attached to the wrong row or
column. This module never compares multisets as a correctness oracle. It
binds each candidate cell to native geometry (a row band AND a lane, both
required — A2) and only convicts a cell when that binding is 1:1 (C3). A cell
whose geometry is ambiguous is never convicted — a false-contradiction is
worse than a missed one.

## Representation

- **Row path**: the row's own stub label, prefixed (tuple, root-first) by any
  panel/section rows above it that carry no numeric tokens. Value-less parent
  rows are kept as first-class rows, not folded into a prefix string.
- **Column header path**: root to leaf, spanning parent first. A spanning
  header is asserted ONLY when native geometry proves it (a header word's
  bbox demonstrably overlaps >= 2 lane intervals) AND the candidate's
  normalised header token at that position matches the native word's text
  (A4). An unproven span is never invented — the column's path stays
  per-lane and its binding is marked unverifiable instead.
- **Empty cells**: first-class slots. ``model_value = None`` matching an
  empty native binding is a MATCH. A native number with no bound candidate
  value is the dropped-digit signal (``native_unbound``, C4). A candidate
  value with no bound native token is the invented-digit signal
  (``model_unbound``, C4's other direction).
- **Row/column binding (A2)**: candidate rows bind to native row bands by an
  anchor/interpolation algorithm, never by row-label text matching (labels
  can collide, e.g. two "Total" rows in different panels) and never by
  multiset equality across the whole table (that is exactly the blind
  oracle this module replaces). Unique per-row numeric-multiset matches are
  anchors; the interval between two anchors (or before the first / after the
  last) binds by order ONLY when the candidate and native counts in that
  interval agree; otherwise every row in that interval is
  ``row_binding_unverifiable`` and nothing in it is convicted. Columns map
  left-to-right only when the candidate's data-column count equals the
  native lane count; otherwise the whole table's column binding is
  unverifiable, and cell-level convictions stop there too — but a lane or
  column with no counterpart under ANY admissible assignment still surfaces
  as ``native_unbound``/``model_unbound`` (see I1 below), and a lane/column
  that DOES have a plausible DP-aligned counterpart is never claimed as a
  binding either way — it is counted ``ambiguous_count`` instead, so a real
  disagreement hidden behind a lane/column mismatch is at least surfaced as
  "not verified" rather than vanishing behind the one flag with no signal
  at all.
- **Uniqueness (C3)**: a native token binds to a cell only when exactly one
  row band and one lane claim it, and neither the band nor the lane is
  ambiguous with a neighbour. Otherwise the token is AMBIGUOUS and
  contributes to no conviction in either direction.
- **Bidirectionality (I1)**: every native row and every candidate row must
  be bound or explicitly reported unbound — a one-sided walk can see a
  dropped native row (round 1's fix) but not an invented candidate row, or
  vice versa. Row-level unbound detection therefore runs unconditionally,
  never gated behind whether column geometry is itself verifiable this
  call: a dropped/invented ROW is a fact about ``_bind_rows``'s whole-row
  multiset anchoring, independent of per-column lane geometry.

Deliberately does NOT use ``native_rows.py::LabeledRow.values`` — it drops
empty cells and parent rows and cannot represent a binding.

Clustering helpers (``_cluster_x_positions``, ``_lane_count_from_words``,
``_well_separated_lanes_in_row``, ``_WELL_SEPARATED_GAP_PT``) and the token
normaliser (``_normalize_numeric_token``, ``is_numeric_token``,
``strip_presentation``) are imported from ``native_verifier.py``, not
reimplemented — ``_normalize_numeric_token`` already preserves decimal
precision (``1.10`` normalises to itself, not ``1.1``; A3 comes for free).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from socr.tables.native_verifier import (
    _WELL_SEPARATED_GAP_PT,
    _cluster_x_positions,
    _lane_count_from_words,
    _normalize_numeric_token,
    _well_separated_lanes_in_row,
    is_numeric_token,
    strip_presentation,
)

# --------------------------------------------------------------------------
# A1 — strict markdown grid parser
# --------------------------------------------------------------------------

# A real separator cell: optional leading/trailing ':' (alignment marker)
# around >= 3 '-'. Anchored so "---" and ":---:" pass, "prose text" does not.
_STRICT_SEP_CELL_RE = re.compile(r"^:?-{3,}:?$")

# A spec-number header token, e.g. "(1)", "(12)" — these are numeric by
# ``_NUM_TOKEN_RE`` but are header decoration, not a data value.
_SPEC_NUMBER_RE = re.compile(r"^\(\d+\)$")


@dataclass(frozen=True)
class Grid:
    """A strictly-parsed markdown table.

    ``header_rows`` and ``rows`` are tuples of raw (stripped) cell text,
    column 0 first. An empty cell is ``""`` in the raw form; callers that
    need the "empty slot" semantics use ``rows`` position, never compact it.
    """

    header_rows: tuple[tuple[str, ...], ...]
    rows: tuple[tuple[str, ...], ...]

    @property
    def n_cols(self) -> int:
        return (
            len(self.header_rows[0])
            if self.header_rows
            else (len(self.rows[0]) if self.rows else 0)
        )


def _split_row(line: str) -> tuple[str, ...]:
    return tuple(c.strip() for c in line.strip().strip("|").split("|"))


def parse_grid(markdown: str) -> Grid | None:
    """Parse *markdown* into a single :class:`Grid`, or ``None``.

    A1: stricter than ``find_table_blocks``. Requires, contiguously:
      - >= 1 header line ('|'-bearing),
      - a genuine separator line immediately after, every cell matching
        ``_STRICT_SEP_CELL_RE``,
      - every row (header and body) has the SAME cell count as the
        separator, and that count is >= 2,
      - >= 1 body row after the separator.

    Pipe-bearing prose with no real separator line parses to ``None`` — it
    is not a phantom grid.

    Only the FIRST such block is returned (this module binds one table per
    call, matching how the winner-selection candidate is scoped to one
    grid page).
    """
    lines = [ln for ln in markdown.splitlines() if ln.strip()]
    pipe_lines = [(i, ln) for i, ln in enumerate(lines) if "|" in ln]
    if not pipe_lines:
        return None

    for sep_pos, (idx, line) in enumerate(pipe_lines):
        cells = _split_row(line)
        if not cells or not all(_STRICT_SEP_CELL_RE.match(c) for c in cells):
            continue
        n_cols = len(cells)
        if n_cols < 2:
            continue
        if sep_pos == 0:
            continue  # no header line before it
        header_idx, header_line = pipe_lines[sep_pos - 1]
        if header_idx != idx - 1:
            continue  # header must be immediately above the separator
        header_cells = _split_row(header_line)
        if len(header_cells) != n_cols:
            continue

        # Walk upward collecting every CONTIGUOUS, equal-width row above the
        # immediate header line too — a multi-level (spanning) header is
        # several such rows stacked root-first before the leaf header row.
        header_block: list[tuple[str, ...]] = [header_cells]
        k = sep_pos - 2
        expected_idx = header_idx - 1
        while k >= 0:
            prev_idx, prev_line = pipe_lines[k]
            if prev_idx != expected_idx:
                break
            prev_cells = _split_row(prev_line)
            if len(prev_cells) != n_cols:
                break
            header_block.append(prev_cells)
            expected_idx -= 1
            k -= 1
        header_block.reverse()

        # Collect contiguous body rows immediately following the separator.
        body_rows: list[tuple[str, ...]] = []
        j = sep_pos + 1
        while j < len(pipe_lines):
            body_idx, body_line = pipe_lines[j]
            if body_idx != idx + 1 + len(body_rows):
                break  # not contiguous with the table block
            body_cells = _split_row(body_line)
            if len(body_cells) != n_cols:
                break
            body_rows.append(body_cells)
            j += 1

        if not body_rows:
            continue

        return Grid(header_rows=tuple(header_block), rows=tuple(body_rows))

    return None


# --------------------------------------------------------------------------
# Native geometry — row bands, lanes, row paths
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _NativeRow:
    y: float
    row_path: tuple[str, ...]
    is_parent: bool  # value-less panel/section row (zero numeric tokens)
    # lane -> (token text, ambiguous) for every numeric token in this row,
    # keyed by the lane index it clustered into (may be ambiguous).
    lane_tokens: dict[int, tuple[str, bool]]
    multiset: Counter  # N2-normalized numeric tokens in this row, for A2 anchoring
    band_ambiguous: bool  # this row's own y-band is not well-separated from a neighbour


def _row_label(words_in_band: list, lane_of: dict[float, int]) -> str:
    """The stub label: every word before the row's first bound data-lane
    token. A numeric row stub (e.g. a year used as the row's own label) is
    not in ``lane_of`` (see :func:`_native_lane_geometry`), so it lands here
    as label text instead of being swallowed as a phantom data column
    (MAJOR 4)."""
    sorted_words = sorted(words_in_band, key=lambda w: w[0])
    first_data_x = next((w[0] for w in sorted_words if lane_of.get(w[0]) is not None), None)
    label_words = [w[4] for w in sorted_words if first_data_x is None or w[0] < first_data_x]
    return " ".join(label_words).strip()


def _assign_bands(words: list) -> tuple[list[float], dict[float, int]]:
    """Cluster all word y0 values into row bands. Reuses the generic clusterer."""
    ys = sorted({round(w[1], 1) for w in words})
    if not ys:
        return [], {}
    centers = _cluster_x_positions(ys)
    y_to_band: dict[float, int] = {}
    for y in ys:
        idx = min(range(len(centers)), key=lambda i: abs(centers[i] - y))
        y_to_band[y] = idx
    return centers, y_to_band


def _band_well_separated(centers: list[float], idx: int) -> bool:
    if len(centers) < 2:
        return True
    neighbours = [abs(centers[idx] - c) for j, c in enumerate(centers) if j != idx]
    return min(neighbours) >= _WELL_SEPARATED_GAP_PT


def _presentation_normalized_for_lanes(words: list) -> list:
    """Return *words* with each numeric token's TEXT replaced by its
    presentation-stripped form (``strip_presentation``); every other word,
    and every word's position, is untouched.

    ``_lane_count_from_words`` (imported from ``native_verifier``, shared
    with its other callers there and never modified here) selects numeric
    tokens with ``_NUM_TOKEN_RE.match`` on the RAW text. Row and header
    parsing in *this* module select numerics with ``is_numeric_token``,
    which strips presentation FIRST. Left to clash, a decorated native
    value — ``**23,126**`` (markdown bold), ``0.05∗∗`` (a Unicode
    significance star), ``$1.10`` (a currency prefix) — is exactly the
    ordinary shape of a typeset econometrics table (GH-103, GH-206), and
    would fail the raw predicate, drop out of ``_lane_count_from_words``
    entirely, and collapse the whole table's lane count to zero: the
    oracle abstains (fails safe, never falsely convicts) but on precisely
    the table shape it exists to check, reintroducing inside this module
    the same drop-class those two issues were filed to eliminate.

    Only TEXT is rewritten; x0/x1/y0/y1 — what lane clustering keys on —
    stay exactly as given, so the returned ``lane_of`` dict remains indexed
    by the caller's original, unmodified x-positions.
    """
    normalized = []
    for wd in words:
        text = wd[4]
        if is_numeric_token(text):
            stripped = strip_presentation(text)
            if stripped != text:
                wd = (wd[0], wd[1], wd[2], wd[3], stripped, *wd[5:])
        normalized.append(wd)
    return normalized


def _native_lane_geometry(
    words: list, n_cand_cols: int | None = None
) -> tuple[int, dict[float, int], list[float]]:
    """Cluster numeric tokens into lanes, then drop any lane that is a STUB
    lane rather than a data lane (MAJOR 4).

    Lane clustering runs on a presentation-normalized copy of *words* (see
    ``_presentation_normalized_for_lanes``), so a decorated native value
    clusters into a lane under the SAME predicate (``is_numeric_token``)
    that row and header parsing use to select numeric tokens elsewhere in
    this module — the raw ``_lane_count_from_words`` predicate alone is
    NOT predicate-consistent with them (it does not know about
    markdown/Unicode presentation marks; ``is_numeric_token`` does).

    A numeric row label (e.g. a year used as the row's own stub, "2020")
    clusters into its own lane exactly like a genuine data column would —
    geometry alone cannot tell them apart. What distinguishes them: a stub
    is, in EVERY row where it appears, the leftmost word of that row —
    nothing, text or number, precedes it. A genuine data lane is preceded
    by the row's label text in at least one row (a row with a blank label
    is not proof either way, since it defers to the OTHER rows sharing that
    lane). Surviving lanes are re-indexed 0..k-1 left to right and their
    centres recomputed from the same x-positions this function clustered
    internally, so this is also the single place lane geometry is derived
    from — callers must not separately recompute lane centres with a
    different numeric predicate.
    """
    raw_count, raw_lane_of = _lane_count_from_words(_presentation_normalized_for_lanes(words))
    if raw_count == 0:
        return 0, {}, []

    _band_centers, y_to_band = _assign_bands(words)
    bands: dict[int, list] = {}
    for w in words:
        bands.setdefault(y_to_band[round(w[1], 1)], []).append(w)

    always_leftmost: dict[int, bool] = {}
    for band_words in bands.values():
        if not band_words:
            continue
        leftmost_x = min(bw[0] for bw in band_words)
        for wd in band_words:
            li = raw_lane_of.get(wd[0])
            if li is None:
                continue
            here = wd[0] == leftmost_x
            always_leftmost[li] = (
                here if li not in always_leftmost else (always_leftmost[li] and here)
            )

    stub_candidates = {li for li, always in always_leftmost.items() if always}

    # "Always leftmost in its own row" is necessary but NOT sufficient
    # evidence of a stub: in a table with no row-label text at all, the
    # leftmost genuine DATA lane is trivially "always leftmost" too (there
    # is nothing to its left in any row), and would be misclassified the
    # same way a real numeric stub is. Only actually remove a lane when
    # doing so is both NECESSARY and SUFFICIENT to reconcile the raw native
    # lane count with the candidate's own data-column count — the same
    # "native geometry proves it AND candidate confirms it" pattern this
    # module already uses for header spans (A4), applied here to lane
    # exclusion instead. If the raw count already matches what the
    # candidate needs, nothing is a stub. If more than one lane would need
    # removing, or exactly one candidate can't be identified, removing
    # nothing (and deferring to the ordinary lane_count-mismatch
    # unverifiable path in ``bind()``) is safer than guessing which lane to
    # drop.
    if n_cand_cols is not None and raw_count == n_cand_cols + 1 and len(stub_candidates) == 1:
        stub_lanes = stub_candidates
    else:
        stub_lanes = set()

    all_centers = _cluster_x_positions(sorted(set(raw_lane_of.keys())))
    surviving_raw = [i for i in range(raw_count) if i not in stub_lanes]
    remap = {raw: new for new, raw in enumerate(surviving_raw)}
    centers = [all_centers[i] for i in surviving_raw]
    lane_of = {x: remap[li] for x, li in raw_lane_of.items() if li in remap}
    return len(surviving_raw), lane_of, centers


def _native_rows(
    words: list, n_cand_cols: int | None = None
) -> tuple[list[_NativeRow], list[float], list[int]]:
    """Parse native words into row bands with row paths and per-lane tokens.

    Header detection stops at the first band that is not "header-like".
    A band is header-like when (a) it has zero numeric tokens, or its
    numeric tokens are all spec-number decoration like "(1)", AND (b) at
    least one of its words sits over the data-lane x-span rather than only
    at the stub (row-label) position. (b) is what keeps a panel/section row
    ("Panel A:", flush against the stub column) from being mistaken for a
    column header merely because it happens to precede the first data row —
    C1 requires those value-less parent rows stay on the data side so their
    row path can prefix the rows beneath them.
    """
    band_centers, y_to_band = _assign_bands(words)
    if not band_centers:
        return [], [], []

    bands: dict[int, list] = {}
    for w in words:
        y_key = round(w[1], 1)
        bands.setdefault(y_to_band[y_key], []).append(w)
    for b in bands.values():
        b.sort(key=lambda w: w[0])

    lane_count, lane_of, lane_centers = _native_lane_geometry(words, n_cand_cols)

    ordered_band_idxs = sorted(bands, key=lambda i: band_centers[i])

    def _is_header_band(band_words: list) -> bool:
        numeric = [w[4] for w in band_words if is_numeric_token(w[4])]
        genuine = [t for t in numeric if not _SPEC_NUMBER_RE.match(strip_presentation(t))]
        if genuine:
            return False
        if not lane_centers:
            return True
        lo, hi = lane_centers[0] - _WELL_SEPARATED_GAP_PT, lane_centers[-1] + _WELL_SEPARATED_GAP_PT
        return any(w[2] > lo and w[0] < hi for w in band_words)

    # --- find where the header region ends ---
    data_start = 0
    for pos, bidx in enumerate(ordered_band_idxs):
        if not _is_header_band(bands[bidx]):
            data_start = pos
            break
    else:
        data_start = len(ordered_band_idxs)

    header_positions = ordered_band_idxs[:data_start]
    data_positions = ordered_band_idxs[data_start:]

    rows: list[_NativeRow] = []
    prefix_stack: list[tuple[float, str]] = []  # (indent x0, label)

    for bidx in data_positions:
        band_words = bands[bidx]
        numeric_words = [w for w in band_words if is_numeric_token(w[4])]
        data_words = [w for w in band_words if lane_of.get(w[0]) is not None]
        label = _row_label(band_words, lane_of)
        band_ambiguous = not _band_well_separated(band_centers, bidx)

        if not data_words and label:
            # value-less parent/panel row: push onto the indent stack and
            # keep it as its own row (C1's "value-less parent rows are kept").
            indent = min(w[0] for w in band_words)
            while prefix_stack and prefix_stack[-1][0] >= indent:
                prefix_stack.pop()
            row_path = tuple(p[1] for p in prefix_stack) + (label,)
            prefix_stack.append((indent, label))
            rows.append(
                _NativeRow(
                    y=band_centers[bidx],
                    row_path=row_path,
                    is_parent=True,
                    lane_tokens={},
                    multiset=Counter(),
                    band_ambiguous=band_ambiguous,
                )
            )
            continue

        row_path = tuple(p[1] for p in prefix_stack) + (label,)

        row_tokens = [(w[0], w[4]) for w in numeric_words]
        row_lane_ids = {lane_of[x] for x, _ in row_tokens if x in lane_of}
        if len(row_lane_ids) >= 2:
            clean_lanes = set(_well_separated_lanes_in_row(row_tokens, lane_of))
        else:
            # `_well_separated_lanes_in_row` needs >= 2 lanes present in the
            # row to judge row-internal jitter and returns [] otherwise — a
            # lone data token isn't ambiguous merely because it's alone in
            # its row. Fall back to whether ITS lane is well separated from
            # its neighbours in the page-wide lane grid instead.
            clean_lanes = {li for li in row_lane_ids if _band_well_separated(lane_centers, li)}

        lane_tokens: dict[int, tuple[str, bool]] = {}
        multiset: Counter = Counter()
        for x, text in row_tokens:
            li = lane_of.get(x)
            if li is None:
                continue  # stub token (e.g. a numeric row label) — not a data cell
            token_ambiguous = band_ambiguous or li not in clean_lanes
            if li in lane_tokens:
                # two tokens landed in the same (band, lane) cell: collision.
                lane_tokens[li] = (lane_tokens[li][0], True)
            else:
                lane_tokens[li] = (text, token_ambiguous)
            multiset[_normalize_numeric_token(text)] += 1

        rows.append(
            _NativeRow(
                y=band_centers[bidx],
                row_path=row_path,
                is_parent=False,
                lane_tokens=lane_tokens,
                multiset=multiset,
                band_ambiguous=band_ambiguous,
            )
        )

    return rows, band_centers, header_positions


def _native_header_words(
    words: list, band_centers: list[float], header_band_idxs: list[int]
) -> list:
    """Return the raw word tuples belonging to header bands, in reading order."""
    out = []
    for w in words:
        y_key = round(w[1], 1)
        idx = min(range(len(band_centers)), key=lambda i: abs(band_centers[i] - y_key))
        if idx in header_band_idxs:
            out.append(w)
    return out


# --------------------------------------------------------------------------
# Column header paths (A4 — spans proven by native geometry + candidate text)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnHeaderPath:
    lane: int
    path: tuple[str, ...]
    spans_lanes: int  # 1 = ordinary; >= 2 = a proven spanning header covers this many lanes
    unverifiable: bool = (
        False  # geometry suggested a span but the candidate text did not confirm it
    )


def _lane_boundaries(lane_centers: list[float]) -> list[tuple[float, float]]:
    """Half-open [lo, hi) x-ranges around each lane centre, midpoint-split."""
    n = len(lane_centers)
    bounds = []
    for i, c in enumerate(lane_centers):
        lo = -float("inf") if i == 0 else (lane_centers[i - 1] + c) / 2
        hi = float("inf") if i == n - 1 else (c + lane_centers[i + 1]) / 2
        bounds.append((lo, hi))
    return bounds


def _lanes_covered(
    word_x0: float, word_x1: float, boundaries: list[tuple[float, float]]
) -> list[int]:
    return [i for i, (lo, hi) in enumerate(boundaries) if word_x1 > lo and word_x0 < hi]


def _candidate_header_confirms_span(grid: Grid, native_text: str, lane0: int, span: int) -> bool:
    """A4 confirmation: candidate has SOME header row whose cell at lane0+1
    (col 0 is the stub) normalises to the same text as the native spanning
    word, with the following (span - 1) candidate cells in that same row
    blank (supporting, not sufficient alone — the native geometry already
    proved the span; this only checks the candidate agrees at that
    position)."""
    if not grid.header_rows:
        return False
    col0 = lane0 + 1
    for header in grid.header_rows:
        if col0 >= len(header):
            continue
        if _norm_header_text(header[col0]) != _norm_header_text(native_text):
            continue
        if all(
            col >= len(header) or header[col].strip() == "" for col in range(col0 + 1, col0 + span)
        ):
            return True
    return False


def _norm_header_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).casefold()


def build_column_header_paths(
    words: list, grid: Grid | None, lane_centers: list[float], header_band_words: list
) -> list[ColumnHeaderPath]:
    """Build one :class:`ColumnHeaderPath` per native lane (A4).

    A span is asserted only when a native header word's bbox demonstrably
    overlaps >= 2 lane intervals AND (if a grid is supplied) the candidate's
    normalised header token at that position matches. Otherwise the lane's
    path is built from whatever native header word solely covers it, and if
    a geometric span was seen but not confirmed, the path is marked
    ``unverifiable`` rather than inventing the merge.
    """
    n_lanes = len(lane_centers)
    if n_lanes == 0:
        return []
    boundaries = _lane_boundaries(lane_centers)

    # Group header words by band (row), top to bottom = root to leaf.
    band_groups: dict[float, list] = {}
    for w in header_band_words:
        band_groups.setdefault(round(w[1], 1), []).append(w)
    ordered_bands = [band_groups[y] for y in sorted(band_groups)]

    per_lane_path: list[list[str]] = [[] for _ in range(n_lanes)]
    per_lane_span: list[int] = [1] * n_lanes
    per_lane_unverifiable: list[bool] = [False] * n_lanes

    for band_words in ordered_bands:
        band_words = sorted(band_words, key=lambda w: w[0])
        # Assign each lane's contribution at this header level.
        level_text: list[str | None] = [None] * n_lanes
        for w in band_words:
            covered = _lanes_covered(w[0], w[2], boundaries)
            if not covered:
                continue
            if len(covered) >= 2:
                proven = grid is None or _candidate_header_confirms_span(
                    grid, w[4], covered[0], len(covered)
                )
                for li in covered:
                    if level_text[li] is not None:
                        continue  # another word already claimed this lane at this level
                    if proven:
                        level_text[li] = w[4]
                        per_lane_span[li] = max(per_lane_span[li], len(covered))
                    else:
                        per_lane_unverifiable[li] = True
            else:
                li = covered[0]
                if level_text[li] is None:
                    level_text[li] = w[4]
        for li in range(n_lanes):
            if level_text[li]:
                per_lane_path[li].append(level_text[li])

    return [
        ColumnHeaderPath(
            lane=li,
            path=tuple(per_lane_path[li]),
            spans_lanes=per_lane_span[li],
            unverifiable=per_lane_unverifiable[li],
        )
        for li in range(n_lanes)
    ]


# --------------------------------------------------------------------------
# A2 — row anchor/interpolation binding
# --------------------------------------------------------------------------


def _candidate_row_multiset(row: tuple[str, ...]) -> Counter:
    c: Counter = Counter()
    for cell in row[1:]:
        for tok in re.split(r"\s+", cell.strip()):
            if tok and is_numeric_token(tok):
                c[_normalize_numeric_token(tok)] += 1
    return c


def _bind_rows(
    native_rows: list[_NativeRow], grid_rows: tuple[tuple[str, ...], ...]
) -> dict[int, int]:
    """Return {candidate_row_idx: native_row_idx} for rows A2 can bind.

    Rows outside the returned mapping are ``row_binding_unverifiable``.
    """
    cand_multisets = [_candidate_row_multiset(r) for r in grid_rows]
    native_multisets = [nr.multiset for nr in native_rows]

    # Index native multisets by their canonical (sorted) tuple form.
    native_by_key: dict[tuple, list[int]] = {}
    for i, ms in enumerate(native_multisets):
        key = tuple(sorted(ms.items()))
        native_by_key.setdefault(key, []).append(i)

    cand_by_key: dict[tuple, list[int]] = {}
    for i, ms in enumerate(cand_multisets):
        key = tuple(sorted(ms.items()))
        cand_by_key.setdefault(key, []).append(i)

    anchors: list[tuple[int, int]] = []  # (cand_idx, native_idx), sorted by cand_idx
    for i, ms in enumerate(cand_multisets):
        if not ms:
            continue  # empty multiset can't anchor uniquely (parent rows, etc.)
        key = tuple(sorted(ms.items()))
        native_matches = native_by_key.get(key, [])
        cand_matches = cand_by_key.get(key, [])
        if len(native_matches) == 1 and len(cand_matches) == 1:
            anchors.append((i, native_matches[0]))

    anchors.sort()
    # Keep only a monotonically increasing subsequence of native indices
    # (a non-monotonic anchor is a false anchor — drop it rather than guess).
    monotonic: list[tuple[int, int]] = []
    last_native = -1
    for cand_idx, native_idx in anchors:
        if native_idx > last_native:
            monotonic.append((cand_idx, native_idx))
            last_native = native_idx

    binding: dict[int, int] = {}
    for cand_idx, native_idx in monotonic:
        binding[cand_idx] = native_idx

    n_cand = len(grid_rows)
    n_native = len(native_rows)
    boundaries = [(-1, -1)] + monotonic + [(n_cand, n_native)]
    for k in range(len(boundaries) - 1):
        c0, nv0 = boundaries[k]
        c1, nv1 = boundaries[k + 1]
        cand_interval = range(c0 + 1, c1)
        native_interval = range(nv0 + 1, nv1)
        if len(cand_interval) == len(native_interval):
            for off, ci in enumerate(cand_interval):
                binding[ci] = native_interval[off]
        # else: leave unbound -> row_binding_unverifiable for this interval

    return binding


# --------------------------------------------------------------------------
# Public result + entry point
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchedCell:
    row_path: tuple[str, ...]
    col_path: tuple[str, ...]
    value: str | None  # None means both sides are the empty slot


@dataclass(frozen=True)
class ContradictedCell:
    row_path: tuple[str, ...]
    col_path: tuple[str, ...]
    native_token: str
    model_token: str | None


@dataclass(frozen=True)
class UnboundCell:
    row_path: tuple[str, ...]
    col_path: tuple[str, ...]
    token: str


@dataclass
class BindingResult:
    matched_cells: list[MatchedCell] = field(default_factory=list)
    contradicted_cells: list[ContradictedCell] = field(default_factory=list)
    native_unbound: list[UnboundCell] = field(default_factory=list)  # dropped-digit signal (C4)
    model_unbound: list[UnboundCell] = field(default_factory=list)  # invented-digit signal (C4)
    ambiguous_count: int = 0
    row_binding_unverifiable: bool = False
    column_binding_unverifiable: bool = False
    column_header_paths: list[ColumnHeaderPath] = field(default_factory=list)

    @property
    def fully_checked(self) -> bool:
        """True when nothing about this binding was left unresolved: every
        lane bound to exactly one candidate column, every native data row
        bound to exactly one candidate row, and no cell's geometry was
        ambiguous. False means some region of the table was never actually
        compared — a different fact from whether the parts that WERE
        compared agreed (MAJOR 2)."""
        return (
            not self.row_binding_unverifiable
            and not self.column_binding_unverifiable
            and self.ambiguous_count == 0
        )

    @property
    def no_known_contradiction(self) -> bool:
        """True when nothing that WAS checked disagreed. Says nothing about
        coverage: a table that was almost entirely unverifiable can still be
        ``no_known_contradiction`` simply because too little of it was
        checkable to find a disagreement in. Read ``structural_agreement``
        (or check this alongside ``fully_checked``) before treating a table
        as verified correct."""
        return not (self.contradicted_cells or self.native_unbound or self.model_unbound)

    @property
    def structural_agreement(self) -> bool:
        """True only when the table was FULLY checkable (every lane, every
        native row, every cell's geometry unambiguous — see
        ``fully_checked``) AND nothing checkable disagreed. An ambiguous or
        unverifiable region is a DIFFERENT fact from disagreement — it means
        "we don't know", not "it matched" — so it makes this False too, on
        purpose: an incompletely-checked table is not a passing table
        (MAJOR 2)."""
        return self.fully_checked and self.no_known_contradiction


def _best_lane_column_map(
    native_rows: list[_NativeRow],
    grid_rows: tuple[tuple[str, ...], ...],
    row_binding: dict[int, int],
    lane_count: int,
    n_cand_cols: int,
) -> dict[int, int]:
    """The single monotone, injective lane -> (0-based data column) map that
    maximises numeric agreement across rows ``_bind_rows`` already anchored.

    Used only when ``lane_count != n_cand_cols`` (``bind`` has already set
    ``column_binding_unverifiable``) to salvage column-level signal from an
    otherwise-abandoned table. Same DP shape as
    ``benchmark.table_exactness._best_lane_column_map`` — map / skip-lane /
    skip-column, monotone and injective, so a genuine column transposition
    is never explained away as a match — but scored from this module's own
    native lane tokens and candidate cells rather than that module's
    ``LabeledRow``, since binding.py's row/lane representation is its own.

    The map is used ONLY to identify which lanes/columns the DP could not
    place anywhere — the ones with no counterpart under ANY admissible
    assignment of the rest — never to claim a value match or contradiction
    for the lanes/columns it DOES place: the table stays
    ``column_binding_unverifiable`` regardless of what this returns.
    """
    if lane_count == 0 or n_cand_cols == 0:
        return {}

    score = [[0] * n_cand_cols for _ in range(lane_count)]
    for cand_idx, native_idx in row_binding.items():
        native_row = native_rows[native_idx]
        if native_row.is_parent:
            continue
        cand_row = grid_rows[cand_idx]
        for lane, (native_text, ambiguous) in native_row.lane_tokens.items():
            if ambiguous or not (0 <= lane < lane_count):
                continue
            for col_idx in range(n_cand_cols):
                col = col_idx + 1
                cand_text = cand_row[col].strip() if col < len(cand_row) else ""
                if not cand_text or not is_numeric_token(cand_text):
                    continue
                if _normalize_numeric_token(native_text) == _normalize_numeric_token(cand_text):
                    score[lane][col_idx] += 1

    dp = [[0] * (n_cand_cols + 1) for _ in range(lane_count + 1)]
    choice = [[""] * (n_cand_cols + 1) for _ in range(lane_count + 1)]
    for i in range(lane_count + 1):
        for j in range(n_cand_cols + 1):
            if i == 0 and j == 0:
                continue
            best, best_choice = -1, ""
            if i > 0 and j > 0:
                candidate = dp[i - 1][j - 1] + score[i - 1][j - 1]
                if candidate > best:
                    best, best_choice = candidate, "map"
            if j > 0 and dp[i][j - 1] > best:
                best, best_choice = dp[i][j - 1], "skip_column"
            if i > 0 and dp[i - 1][j] > best:
                best, best_choice = dp[i - 1][j], "skip_lane"
            dp[i][j] = best
            choice[i][j] = best_choice

    mapping: dict[int, int] = {}
    i, j = lane_count, n_cand_cols
    while i > 0 or j > 0:
        move = choice[i][j]
        if move == "map":
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif move == "skip_column":
            j -= 1
        else:
            i -= 1
    return mapping


def bind(words: list, markdown: str) -> BindingResult:
    """Bind *markdown*'s candidate grid to the native geometry in *words*.

    Never raises on malformed input: a markdown block that fails the A1
    strict parse, or a page with no numeric lanes, binds nothing and returns
    an (empty) :class:`BindingResult` — an absence of evidence, not a
    conviction of either side.
    """
    result = BindingResult()

    grid = parse_grid(markdown)
    if grid is None:
        return result

    n_cand_cols = grid.n_cols - 1  # exclude the stub column

    native_rows, band_centers, header_band_idxs = _native_rows(words, n_cand_cols)
    lane_count, _lane_of, lane_centers = _native_lane_geometry(words, n_cand_cols)

    header_words = (
        _native_header_words(words, band_centers, header_band_idxs) if header_band_idxs else []
    )
    result.column_header_paths = build_column_header_paths(words, grid, lane_centers, header_words)

    # I1 BIDIRECTIONALITY: row-level binding and its unbound-row signals run
    # regardless of whether column geometry (lane_count vs n_cand_cols) is
    # even usable this call. `_bind_rows` anchors rows from whole-row
    # multisets, independent of lane geometry, so gating this behind the
    # lane/column check (an early `return` used to) silently swallowed
    # every row-level drop/invention signal whenever column geometry was
    # ALSO unverifiable — this is exactly what HIGH 1 and HIGH 2 reported.
    row_binding = _bind_rows(native_rows, grid.rows)
    if len(row_binding) != len(grid.rows):
        result.row_binding_unverifiable = True

    # BLOCKING 1: a native DATA row that no candidate row ever bound to is not
    # merely "unverifiable" in the abstract — it is C4's dropped-digit signal
    # for the whole row. `_bind_rows` only tracks candidate coverage; a
    # dropped native row can still leave every candidate row bound (e.g. the
    # anchors either side of the gap still line up), so it must be checked
    # separately here rather than inferred from `len(row_binding)`.
    bound_native_idxs = set(row_binding.values())
    unbound_native_rows = [
        (idx, nr)
        for idx, nr in enumerate(native_rows)
        if not nr.is_parent and idx not in bound_native_idxs
    ]
    if unbound_native_rows:
        result.row_binding_unverifiable = True

    header_paths_by_lane = {chp.lane: chp for chp in result.column_header_paths}

    for idx, nr in unbound_native_rows:
        # MEDIUM 3: the row itself having no candidate counterpart at all is
        # a STRONGER, more specific fact than a token's own in-row lane
        # ambiguity (which exists to guard the per-cell walk below, where a
        # wrong lane assignment could misattribute a value to the wrong
        # column) — do not let `ambiguous` demote this to a vague
        # `ambiguous_count` and hide that the whole row was dropped.
        for lane, (text, _ambiguous) in nr.lane_tokens.items():
            chp = header_paths_by_lane.get(lane)
            col_path = chp.path if chp else (str(lane),)
            result.native_unbound.append(
                UnboundCell(row_path=nr.row_path, col_path=col_path, token=text)
            )

    # HIGH 2: the candidate-side mirror of BLOCKING 1. An invented candidate
    # row that `_bind_rows` could not anchor to anything never enters the
    # per-cell walk below (it isn't a key in `row_binding`), so its values
    # must be reported here or they vanish — C4's invented-digit signal,
    # dropped instead of surfaced.
    for cand_idx, cand_row in enumerate(grid.rows):
        if cand_idx in row_binding:
            continue
        row_path = (cand_row[0],) if cand_row and cand_row[0] else ()
        for col in range(1, len(cand_row)):
            model_value = cand_row[col].strip()
            if not model_value or not is_numeric_token(model_value):
                continue
            lane = col - 1
            chp = header_paths_by_lane.get(lane)
            col_path = chp.path if chp else (str(lane),)
            result.model_unbound.append(
                UnboundCell(row_path=row_path, col_path=col_path, token=model_value)
            )

    if lane_count == 0 or n_cand_cols != lane_count:
        result.column_binding_unverifiable = True

        # HIGH 1: column geometry itself is unverifiable, but that is not
        # licence to drop every cell signal for the whole table — only to
        # stop CLAIMING a binding for it. `_best_lane_column_map` salvages
        # what row_binding already proves: for each row we DO know binds,
        # any lane/column the map could not place anywhere (under ANY
        # admissible assignment) is unbound content, not an unknown.
        lane_to_col = _best_lane_column_map(
            native_rows, grid.rows, row_binding, lane_count, n_cand_cols
        )
        mapped_lanes = set(lane_to_col.keys())
        mapped_cols = set(lane_to_col.values())

        for cand_idx, native_idx in row_binding.items():
            native_row = native_rows[native_idx]
            if native_row.is_parent:
                continue
            cand_row = grid.rows[cand_idx]

            # I1 follow-up: a lane/column the DP maps has a plausible
            # counterpart, so it is not the dropped/invented-digit signal
            # native_unbound/model_unbound exist for -- but leaving it
            # completely unreported was itself an unreported third state,
            # no better than the silence this branch exists to fix. Count
            # it as ambiguous -- the same "known geometry, not confidently
            # convictable either way" bucket C3 already uses for exactly
            # this shape of uncertainty -- once per row per mapped pair (one
            # physical cell shared by both sides of I1), so a disagreement
            # hidden behind a lane/column mismatch surfaces as an honest
            # "not verified" rather than vanishing with no signal at all.
            for lane, col_idx in lane_to_col.items():
                col = col_idx + 1
                cand_text = cand_row[col].strip() if col < len(cand_row) else ""
                if lane in native_row.lane_tokens or (cand_text and is_numeric_token(cand_text)):
                    result.ambiguous_count += 1

            for lane, (text, ambiguous) in native_row.lane_tokens.items():
                if lane in mapped_lanes:
                    continue
                if ambiguous:
                    result.ambiguous_count += 1
                    continue
                chp = header_paths_by_lane.get(lane)
                col_path = chp.path if chp else (str(lane),)
                result.native_unbound.append(
                    UnboundCell(row_path=native_row.row_path, col_path=col_path, token=text)
                )

            for col_idx in range(n_cand_cols):
                if col_idx in mapped_cols:
                    continue
                col = col_idx + 1
                cand_text = cand_row[col].strip() if col < len(cand_row) else ""
                if not cand_text or not is_numeric_token(cand_text):
                    continue
                chp = header_paths_by_lane.get(col_idx)
                col_path = chp.path if chp else (str(col_idx),)
                result.model_unbound.append(
                    UnboundCell(row_path=native_row.row_path, col_path=col_path, token=cand_text)
                )

        return result

    for cand_idx, cand_row in enumerate(grid.rows):
        if cand_idx not in row_binding:
            continue
        native_idx = row_binding[cand_idx]
        native_row = native_rows[native_idx]
        if native_row.is_parent:
            continue

        for lane in range(lane_count):
            col = lane + 1
            model_raw = cand_row[col] if col < len(cand_row) else ""
            model_value = model_raw.strip() or None
            chp = header_paths_by_lane.get(lane)
            col_path = chp.path if chp else (str(lane),)

            token_entry = native_row.lane_tokens.get(lane)
            if token_entry is None:
                native_value = None
                native_ambiguous = False
            else:
                native_value, native_ambiguous = token_entry

            if native_ambiguous:
                result.ambiguous_count += 1
                continue

            if native_value is None and model_value is None:
                result.matched_cells.append(
                    MatchedCell(row_path=native_row.row_path, col_path=col_path, value=None)
                )
            elif native_value is None and model_value is not None:
                result.model_unbound.append(
                    UnboundCell(row_path=native_row.row_path, col_path=col_path, token=model_value)
                )
            elif native_value is not None and model_value is None:
                result.native_unbound.append(
                    UnboundCell(row_path=native_row.row_path, col_path=col_path, token=native_value)
                )
            else:
                if not is_numeric_token(model_value):
                    # candidate cell isn't a numeric token at all: treat as a
                    # value mismatch against a native number.
                    result.contradicted_cells.append(
                        ContradictedCell(
                            row_path=native_row.row_path,
                            col_path=col_path,
                            native_token=native_value,
                            model_token=model_value,
                        )
                    )
                    continue
                if _normalize_numeric_token(native_value) == _normalize_numeric_token(model_value):
                    result.matched_cells.append(
                        MatchedCell(
                            row_path=native_row.row_path, col_path=col_path, value=native_value
                        )
                    )
                else:
                    result.contradicted_cells.append(
                        ContradictedCell(
                            row_path=native_row.row_path,
                            col_path=col_path,
                            native_token=native_value,
                            model_token=model_value,
                        )
                    )

    return result
