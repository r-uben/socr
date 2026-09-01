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
  ``row_binding_unverifiable`` and nothing in it is convicted. Once numeric
  content and order have established a row binding, the candidate's stub is
  verified against that native row's own label; labels verify an existing
  binding but never choose one. Columns map left-to-right only when the
  candidate's data-column count equals the native lane count; otherwise the
  whole table's column binding is unverifiable, and cell-level convictions
  stop there too — but a lane or column with no counterpart under ANY
  admissible assignment still surfaces as ``native_unbound``/``model_unbound``
  (see I1 below), and a lane/column
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

from socr.tables.native_rows import normalize_label
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


def _candidate_data_column_indices(grid: Grid) -> tuple[int, ...]:
    """Return original indexes for candidate columns with genuine numeric data.

    The rowizer emits a fixed lane grid before ``_clean_grid`` sees it.  A
    lane can consequently remain in the markdown when its header is populated
    but every body cell is empty; spec-number decorations such as ``(1)`` have
    the same header-only shape.  Such a column has no numeric binding claim.
    Projecting it out keeps the binder's column space aligned with the
    candidate's numeric data space without selecting a native lane by value.

    ``_project_candidate_data_columns`` and ``bind()``'s column remap both
    consume this tuple so the projected grid and index mapping cannot drift.
    """
    numeric_columns_by_row: list[set[int]] = []
    for column in range(1, grid.n_cols):
        for row_number, row in enumerate(grid.rows):
            if len(numeric_columns_by_row) <= row_number:
                numeric_columns_by_row.append(set())
            if column >= len(row):
                continue
            if any(
                is_numeric_token(token) and not _SPEC_NUMBER_RE.match(strip_presentation(token))
                for token in re.split(r"\s+", row[column].strip())
                if token
            ):
                numeric_columns_by_row[row_number].add(column)

    if not numeric_columns_by_row:
        return tuple()
    widest_rows = max(len(columns) for columns in numeric_columns_by_row)
    data_columns = sorted(
        set().union(*(columns for columns in numeric_columns_by_row if len(columns) == widest_rows))
    )
    return tuple(data_columns)


def _project_candidate_data_columns(grid: Grid) -> Grid:
    """Project *grid* down to stub plus the numeric data columns from the helper."""
    data_columns = _candidate_data_column_indices(grid)
    if not data_columns:
        return grid
    keep = (0, *data_columns)
    return Grid(
        header_rows=tuple(tuple(row[column] for column in keep) for row in grid.header_rows),
        rows=tuple(tuple(row[column] for column in keep) for row in grid.rows),
    )


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
    # GH-367: native word bboxes so adjudication can crop the paint bind()
    # compared against, without re-deriving geometry. Diagnostic only —
    # never consulted by conviction logic.
    lane_bboxes: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    label_bbox: tuple[float, float, float, float] | None = None


def _union_word_bbox(words: list) -> tuple[float, float, float, float] | None:
    if not words:
        return None
    return (
        min(w[0] for w in words),
        min(w[1] for w in words),
        max(w[2] for w in words),
        max(w[3] for w in words),
    )


def _row_label(words_in_band: list, lane_of: dict[float, int]) -> str:
    """The stub label: every word before the row's first bound data-lane
    token. A numeric row stub (e.g. a year used as the row's own label) is
    not in ``lane_of`` (see :func:`_native_lane_geometry`), so it lands here
    as label text instead of being swallowed as a phantom data column
    (MAJOR 4)."""
    return _row_label_and_bbox(words_in_band, lane_of)[0]


def _row_label_and_bbox(
    words_in_band: list, lane_of: dict[float, int]
) -> tuple[str, tuple[float, float, float, float] | None]:
    sorted_words = sorted(words_in_band, key=lambda w: w[0])
    first_data_x = next((w[0] for w in sorted_words if lane_of.get(w[0]) is not None), None)
    label_words = [w for w in sorted_words if first_data_x is None or w[0] < first_data_x]
    label = " ".join(w[4] for w in label_words).strip()
    return label, _union_word_bbox(label_words)


def _assign_bands(words: list) -> tuple[list[float], dict[float, int]]:
    """Assign rowizer-compatible y groups without chaining adjacent rows.

    ``rowize_from_word_list`` uses ``round(y0)`` as its row key.  Keep that
    exact partition here: unlike x-lane clustering, it cannot make a run of
    nearby printed rows collapse into one band.

    A superscript or marker can have a different y0 from the number it
    annotates.  Such a numeric-free group is folded only when its PyMuPDF
    ``(block_no, line_no)`` metadata points to exactly one numeric-bearing
    y-group; exact bbox intersection is only a corroborating guard against
    synthetic or stale metadata on distant prose. No distance tolerance is
    used as row evidence.
    """
    rows_by_y: dict[int, list] = {}
    for word in words:
        rows_by_y.setdefault(round(word[1]), []).append(word)

    if not rows_by_y:
        return [], {}

    numeric_y_keys = {
        y_key
        for y_key, row_words in rows_by_y.items()
        if any(is_numeric_token(word[4]) for word in row_words)
    }
    numeric_words_by_y = {
        y_key: [word for word in rows_by_y[y_key] if is_numeric_token(word[4])]
        for y_key in numeric_y_keys
    }

    # A metadata line may contain words in more than one y-group.  Retain all
    # such groups so that folding is allowed only when the line identity has a
    # unique numeric-bearing destination.
    line_to_numeric_groups: dict[tuple[object, object], set[int]] = {}
    for y_key in numeric_y_keys:
        for word in rows_by_y[y_key]:
            line_key = (word[5], word[6])
            line_to_numeric_groups.setdefault(line_key, set()).add(y_key)

    y_to_group_key = {y_key: y_key for y_key in rows_by_y}
    for y_key, row_words in rows_by_y.items():
        if y_key in numeric_y_keys:
            continue
        destinations = set()
        for word in row_words:
            for destination in line_to_numeric_groups.get((word[5], word[6]), ()):
                # A displaced marker is part of the same printed line when
                # its extracted box intersects the numeric word's box. This
                # exact geometry guard keeps default/synthetic metadata on
                # distant headers and panel rows from being treated as line
                # evidence; no proximity radius is introduced.
                if any(
                    word[1] < numeric_word[3] and numeric_word[1] < word[3]
                    for numeric_word in numeric_words_by_y[destination]
                ):
                    destinations.add(destination)
        if len(destinations) == 1:
            y_to_group_key[y_key] = destinations.pop()

    group_keys = sorted(set(y_to_group_key.values()))
    group_to_band = {group_key: idx for idx, group_key in enumerate(group_keys)}
    y_to_band = {y_key: group_to_band[group_key] for y_key, group_key in y_to_group_key.items()}
    return [float(group_key) for group_key in group_keys], y_to_band


def _lane_well_separated(centers: list[float], idx: int) -> bool:
    if len(centers) < 2:
        return True
    neighbours = [abs(centers[idx] - c) for j, c in enumerate(centers) if j != idx]
    return min(neighbours) >= _WELL_SEPARATED_GAP_PT


def _ambiguous_bands(bands: dict[int, list], ordered_band_idxs: list[int]) -> set[int]:
    """Return bands whose word extents overlap an adjacent band's extent.

    Row ambiguity is a property of the extracted word boxes, not of their
    center pitch.  Compare only consecutive bands in reading order and use a
    strict overlap test, so boxes that merely touch are still unambiguous.
    Numeric-bearing bands use their numeric words for the extent: a displaced
    annotation marker folded into its owning line must not extend that line's
    binding band into a neighbouring row.  A value-less band uses all of its
    words because it has no numeric binding extent.
    """
    extents = {}
    for bidx in ordered_band_idxs:
        numeric_words = [word for word in bands[bidx] if is_numeric_token(word[4])]
        extent_words = numeric_words or bands[bidx]
        extents[bidx] = (
            min(word[1] for word in extent_words),
            max(word[3] for word in extent_words),
        )
    ambiguous: set[int] = set()
    for left_idx, right_idx in zip(ordered_band_idxs, ordered_band_idxs[1:]):
        left_y0, left_y1 = extents[left_idx]
        right_y0, right_y1 = extents[right_idx]
        if min(left_y1, right_y1) > max(left_y0, right_y0):
            ambiguous.update((left_idx, right_idx))
    return ambiguous


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
        bands.setdefault(y_to_band[round(w[1])], []).append(w)

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

    # The rowizer discovers lanes from every numeric word in its segment, but
    # its emitted grid is subsequently cleaned.  A numeric word that occurs in
    # an isolated band (for example a page number in a captured running head)
    # therefore has no data-cell counterpart: it cannot make a rowizer data
    # column.  Keep only lanes that participate in a band with at least two
    # numeric lanes.  The multi-lane requirement is structural evidence of a
    # table row, not a distance or density threshold; if no such band exists,
    # retain all lanes so a legitimate one-column table still binds.
    data_lanes: set[int] = set(range(raw_count))
    if n_cand_cols is not None and raw_count > n_cand_cols:
        data_lanes = set()
        for band_words in bands.values():
            band_lanes = {
                raw_lane_of[word[0]]
                for word in band_words
                if is_numeric_token(word[4]) and word[0] in raw_lane_of
            }
            if len(band_lanes) > 1:
                data_lanes.update(band_lanes)
        if not data_lanes:
            data_lanes = set(range(raw_count))

    # Lanes outside a multi-lane data band are the native equivalent of the
    # all-empty columns removed by ``_clean_grid``.  This is geometry-only
    # exclusion: candidate width may confirm the resulting count, but never
    # identifies one lane among several otherwise plausible lanes.
    surviving_raw = [i for i in range(raw_count) if i in data_lanes]
    if n_cand_cols is not None and len(surviving_raw) > n_cand_cols:
        surviving_set = set(surviving_raw)
        remaining_stub_candidates = stub_candidates & surviving_set
        if len(surviving_raw) == n_cand_cols + 1 and len(remaining_stub_candidates) == 1:
            surviving_raw = [i for i in surviving_raw if i not in remaining_stub_candidates]

    all_centers = _cluster_x_positions(sorted(set(raw_lane_of.keys())))
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
        y_key = round(w[1])
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
    ambiguous_bands = _ambiguous_bands(bands, ordered_band_idxs)

    rows: list[_NativeRow] = []
    prefix_stack: list[tuple[float, str]] = []  # (indent x0, label)

    for bidx in data_positions:
        band_words = bands[bidx]
        numeric_words = [w for w in band_words if is_numeric_token(w[4])]
        data_words = [w for w in band_words if lane_of.get(w[0]) is not None]
        label, label_bbox = _row_label_and_bbox(band_words, lane_of)
        band_ambiguous = bidx in ambiguous_bands

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
                    lane_bboxes={},
                    label_bbox=label_bbox,
                )
            )
            continue

        row_path = tuple(p[1] for p in prefix_stack) + (label,)

        row_tokens = [(w[0], w[4], w) for w in numeric_words]
        row_lane_ids = {lane_of[x] for x, _, _ in row_tokens if x in lane_of}
        if len(row_lane_ids) >= 2:
            clean_lanes = set(
                _well_separated_lanes_in_row([(x, text) for x, text, _ in row_tokens], lane_of)
            )
        else:
            # `_well_separated_lanes_in_row` needs >= 2 lanes present in the
            # row to judge row-internal jitter and returns [] otherwise — a
            # lone data token isn't ambiguous merely because it's alone in
            # its row. Fall back to whether ITS lane is well separated from
            # its neighbours in the page-wide lane grid instead.
            clean_lanes = {li for li in row_lane_ids if _lane_well_separated(lane_centers, li)}

        lane_tokens: dict[int, tuple[str, bool]] = {}
        lane_bboxes: dict[int, tuple[float, float, float, float]] = {}
        multiset: Counter = Counter()
        for x, text, word in row_tokens:
            li = lane_of.get(x)
            if li is None:
                continue  # stub token (e.g. a numeric row label) — not a data cell
            token_ambiguous = band_ambiguous or li not in clean_lanes
            if li in lane_tokens:
                # two tokens landed in the same (band, lane) cell: collision.
                lane_tokens[li] = (lane_tokens[li][0], True)
            else:
                lane_tokens[li] = (text, token_ambiguous)
                lane_bboxes[li] = (word[0], word[1], word[2], word[3])
            multiset[_normalize_numeric_token(text)] += 1

        rows.append(
            _NativeRow(
                y=band_centers[bidx],
                row_path=row_path,
                is_parent=False,
                lane_tokens=lane_tokens,
                multiset=multiset,
                band_ambiguous=band_ambiguous,
                lane_bboxes=lane_bboxes,
                label_bbox=label_bbox,
            )
        )

    return rows, band_centers, header_positions


def _native_header_words(
    words: list, band_centers: list[float], header_band_idxs: list[int]
) -> list:
    """Return the raw word tuples belonging to header bands, in reading order."""
    out = []
    for w in words:
        y_key = round(w[1])
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
        band_groups.setdefault(round(w[1]), []).append(w)
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

    Numeric rows are anchored and interpolated in compressed sequences that
    omit value-less rows.  Once an interval's numeric order is established,
    value-less rows in the corresponding original-row interval may be paired
    by order only when both sides have the same number of value-less rows.
    This keeps an empty candidate row from being used as padding for a native
    numeric row (or vice versa).

    Rows outside the returned mapping are reported by :func:`bind`; whether
    that makes row binding unverifiable is determined from numeric-row
    coverage, while the two value-less row populations are exposed as
    content-free counters.
    """
    cand_multisets = [_candidate_row_multiset(r) for r in grid_rows]
    native_multisets = [nr.multiset for nr in native_rows]

    # Compress both sequences before doing any anchoring or interpolation.
    # An empty multiset is a panel, units, note, or other value-less row; it
    # has no numeric identity with which to establish an interval boundary.
    cand_numeric_idxs = [idx for idx, multiset in enumerate(cand_multisets) if multiset]
    native_numeric_idxs = [idx for idx, multiset in enumerate(native_multisets) if multiset]
    cand_numeric_multisets = [cand_multisets[idx] for idx in cand_numeric_idxs]
    native_numeric_multisets = [native_multisets[idx] for idx in native_numeric_idxs]

    # Index numeric multisets by their canonical (sorted) tuple form.
    native_by_key: dict[tuple, list[int]] = {}
    for i, ms in enumerate(native_numeric_multisets):
        key = tuple(sorted(ms.items()))
        native_by_key.setdefault(key, []).append(i)

    cand_by_key: dict[tuple, list[int]] = {}
    for i, ms in enumerate(cand_numeric_multisets):
        key = tuple(sorted(ms.items()))
        cand_by_key.setdefault(key, []).append(i)

    # Coordinates here are positions in the compressed numeric sequences,
    # not positions in the original row lists.  This is the key distinction:
    # an inserted candidate units row must not change which numeric rows are
    # in the interval between two anchors.
    anchors: list[tuple[int, int]] = []  # (compressed cand idx, compressed native idx)
    for i, ms in enumerate(cand_numeric_multisets):
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

    binding: dict[int, int] = {
        cand_numeric_idxs[cand_idx]: native_numeric_idxs[native_idx]
        for cand_idx, native_idx in monotonic
    }

    n_cand_numeric = len(cand_numeric_idxs)
    n_native_numeric = len(native_numeric_idxs)
    boundaries = [(-1, -1)] + monotonic + [(n_cand_numeric, n_native_numeric)]
    for k in range(len(boundaries) - 1):
        c0, nv0 = boundaries[k]
        c1, nv1 = boundaries[k + 1]
        cand_numeric_interval = range(c0 + 1, c1)
        native_numeric_interval = range(nv0 + 1, nv1)
        numeric_interval_is_ordered = len(cand_numeric_interval) == len(native_numeric_interval)
        if numeric_interval_is_ordered:
            for off, ci in enumerate(cand_numeric_interval):
                binding[cand_numeric_idxs[ci]] = native_numeric_idxs[native_numeric_interval[off]]

        # The compressed interval's endpoints are numeric rows in the
        # original sequences.  The slices between those endpoints therefore
        # contain only value-less rows.  Pairing is allowed only after the
        # numeric interval itself has an order-preserving interpretation and
        # the value-less populations have the same cardinality.  In
        # particular, an empty candidate row can never consume a native row
        # that carries numbers just to make the original row counts equal.
        cand_start = -1 if c0 < 0 else cand_numeric_idxs[c0]
        cand_end = len(grid_rows) if c1 == n_cand_numeric else cand_numeric_idxs[c1]
        native_start = -1 if nv0 < 0 else native_numeric_idxs[nv0]
        native_end = len(native_rows) if nv1 == n_native_numeric else native_numeric_idxs[nv1]
        cand_valueless_interval = [
            idx for idx in range(cand_start + 1, cand_end) if not cand_multisets[idx]
        ]
        native_valueless_interval = [
            idx for idx in range(native_start + 1, native_end) if not native_multisets[idx]
        ]
        if numeric_interval_is_ordered and len(cand_valueless_interval) == len(
            native_valueless_interval
        ):
            for cand_idx, native_idx in zip(cand_valueless_interval, native_valueless_interval):
                binding[cand_idx] = native_idx
        # Otherwise the value-less rows stay unbound and are counted by bind().

    # A candidate parent can contain invented values even though the native
    # parent is value-less.  Preserve the existing parent-row invention
    # diagnostic for that narrow, label-confirmed shape when the compressed
    # numeric sequences have different lengths because of the candidate's
    # lane layout.  This is not a general row-count equalisation fallback:
    # every pair must be label-compatible, and an empty candidate row is
    # explicitly never allowed to consume a numeric native row.
    remaining_candidates = [idx for idx in range(len(grid_rows)) if idx not in binding]
    remaining_native = [idx for idx in range(len(native_rows)) if idx not in binding.values()]
    if len(remaining_candidates) == len(remaining_native):
        fallback_pairs = list(zip(remaining_candidates, remaining_native))

        def _fallback_pair_allowed(cand_idx: int, native_idx: int) -> bool:
            candidate_multiset = cand_multisets[cand_idx]
            native_row = native_rows[native_idx]
            candidate_label = grid_rows[cand_idx][0].strip()
            native_label = native_row.row_path[-1].strip() if native_row.row_path else ""
            labels_match = bool(candidate_label and native_label) and normalize_label(
                candidate_label
            ) == normalize_label(native_label)
            if not labels_match:
                return False
            if not candidate_multiset:
                # In particular, do not pair a value-less candidate with a
                # numeric native merely because the unresolved lists align.
                return bool(not native_row.multiset)
            return bool(native_row.multiset or native_row.is_parent)

        has_parent_invention = any(
            cand_multisets[cand_idx]
            and not native_rows[native_idx].multiset
            and native_rows[native_idx].is_parent
            for cand_idx, native_idx in fallback_pairs
        )
        if has_parent_invention and all(
            _fallback_pair_allowed(cand_idx, native_idx) for cand_idx, native_idx in fallback_pairs
        ):
            for cand_idx, native_idx in fallback_pairs:
                binding[cand_idx] = native_idx

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
    native_bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class RowLabelContradiction:
    row_path: tuple[str, ...]
    candidate_label: str
    native_bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class UnboundCell:
    row_path: tuple[str, ...]
    col_path: tuple[str, ...]
    token: str


@dataclass
class BindingResult:
    """Cell, row, and coverage observations from one binding attempt.

    ``candidate_valueless_unbound`` and ``native_valueless_unbound`` count
    unmatched rows whose numeric multiset is empty on the respective side.
    They are diagnostic counters and do not make complete numeric coverage
    unverifiable. ``row_labels_checked`` counts every row in the binding map,
    including value-less parent rows; all three fields are populated directly
    so reporting callers need no fallback attribute handling.
    """

    matched_cells: list[MatchedCell] = field(default_factory=list)
    contradicted_cells: list[ContradictedCell] = field(default_factory=list)
    row_label_contradictions: list[RowLabelContradiction] = field(default_factory=list)
    native_unbound: list[UnboundCell] = field(default_factory=list)  # dropped-digit signal (C4)
    model_unbound: list[UnboundCell] = field(default_factory=list)  # invented-digit signal (C4)
    ambiguous_count: int = 0
    row_binding_unverifiable: bool = True
    row_label_unverifiable: bool = False
    column_binding_unverifiable: bool = True
    column_header_paths: list[ColumnHeaderPath] = field(default_factory=list)
    # Content-free row coverage diagnostics.  Keep these as explicit counters
    # so callers do not have to infer them from optional result attributes.
    candidate_valueless_unbound: int = 0
    native_valueless_unbound: int = 0
    row_labels_checked: int = 0

    @property
    def fully_checked(self) -> bool:
        """True when nothing about this binding was left unresolved: every
        lane bound to exactly one candidate column, every native data row
        bound to exactly one candidate row, every bound row label checked,
        and no cell's geometry was ambiguous. False means some region of the
        table was never actually compared — a different fact from whether the parts that WERE
        compared agreed (MAJOR 2)."""
        return (
            not self.row_binding_unverifiable
            and not self.row_label_unverifiable
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
        return not (
            self.contradicted_cells
            or self.row_label_contradictions
            or self.native_unbound
            or self.model_unbound
        )

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


def _record_inventions_on_parent_row(
    result: BindingResult,
    native_row: _NativeRow,
    cand_row: tuple[str, ...],
    n_cand_cols: int,
    header_paths_by_lane: dict[int, ColumnHeaderPath],
    col_to_lane: dict[int, int],
) -> None:
    """Record invented digits on a candidate row bound to a native parent.

    A native parent (panel/section heading) has no numeric cells. Binding
    the candidate row to it used to skip both cell walks, so any numbers
    the model wrote on that row vanished — they were in ``row_binding``
    (so HIGH 2 did not report them) and then ``continue``'d (so the walks
    did not either). Empty candidate cells stay a no-op: a genuine empty
    heading row is not an invention.

    ``col_to_lane`` maps a candidate data-column index to a native lane.
    At the equal-count walk that map is the identity; at the salvage walk
    it is the inverse of ``lane_to_col``. Looking up
    ``header_paths_by_lane`` with a candidate column index is only correct
    when those two index spaces coincide — they do not under salvage.
    A column with no native lane (or a lane with no header path) reports
    an empty path: a bare index is not a header.
    """
    for col_idx in range(n_cand_cols):
        col = col_idx + 1
        cand_text = cand_row[col].strip() if col < len(cand_row) else ""
        if not cand_text or not is_numeric_token(cand_text):
            continue
        lane = col_to_lane.get(col_idx)
        chp = header_paths_by_lane.get(lane) if lane is not None else None
        col_path = chp.path if chp is not None else ()
        result.model_unbound.append(
            UnboundCell(row_path=native_row.row_path, col_path=col_path, token=cand_text)
        )


def _words_in_region(words: list, region: tuple | None) -> list:
    """Filter *words* to those whose top-left corner lies inside *region*.

    GH-330. ``bind`` was only ever called with a whole page's words, so on a page
    with prose above the table and notes below it, lane clustering ran over text
    that is not in the table at all — which is why column binding was unverifiable
    on every real page measured. Every native table region already arrives as a
    ``(rect, markdown)`` pair, so the rect was available all along and simply never
    passed in.

    Top-left containment (not intersection) matches how ``extract_structured``
    assigns a word to a region, so the two agree on which words belong to a table.
    Kept here rather than imported so ``binding`` stays free of ``fitz``.

    ``region=None`` returns *words* unchanged — byte-for-byte the old behaviour.
    """
    if region is None:
        return words
    try:
        x0, y0, x1, y1 = (float(v) for v in region)
    except (TypeError, ValueError):
        return words  # a malformed region is an absence of scoping, not a conviction
    if not (x0 <= x1 and y0 <= y1):
        return words
    return [w for w in words if x0 <= w[0] <= x1 and y0 <= w[1] <= y1]


def bind(words: list, markdown: str, *, region: tuple | None = None) -> BindingResult:
    """Bind *markdown*'s candidate grid to the native geometry in *words*.

    Never raises on malformed input: a markdown block that fails the A1
    strict parse, or a page with no numeric lanes, binds nothing and returns
    an (empty) :class:`BindingResult` — an absence of evidence, not a
    conviction of either side.

    *region*, when given, is the candidate's own ``(x0, y0, x1, y1)`` extent; words
    outside it are dropped before any geometry is computed (GH-330). Omitting it is
    the unscoped whole-page fallback, whose column binding is expected to be
    unverifiable on any page that carries text outside the table.
    """
    words = _words_in_region(words, region)
    result = BindingResult()

    grid = parse_grid(markdown)
    if grid is None:
        return result

    candidate_grid = grid
    physical_n_cand_cols = candidate_grid.n_cols - 1
    raw_lane_count = _lane_count_from_words(_presentation_normalized_for_lanes(words))[0]
    if raw_lane_count < physical_n_cand_cols:
        candidate_data_columns = _candidate_data_column_indices(candidate_grid)
        grid = _project_candidate_data_columns(candidate_grid)
        if not candidate_data_columns:
            candidate_data_columns = tuple(range(1, physical_n_cand_cols + 1))
            grid = candidate_grid
    else:
        candidate_data_columns = tuple(range(1, physical_n_cand_cols + 1))
        grid = candidate_grid
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
    row_binding = _bind_rows(native_rows, candidate_grid.rows)
    bound_candidate_idxs = set(row_binding)
    bound_native_idxs = set(row_binding.values())
    result.candidate_valueless_unbound = sum(
        1
        for idx, row in enumerate(candidate_grid.rows)
        if idx not in bound_candidate_idxs and not _candidate_row_multiset(row)
    )
    result.native_valueless_unbound = sum(
        1
        for idx, native_row in enumerate(native_rows)
        if idx not in bound_native_idxs and not native_row.multiset
    )
    candidate_numeric_idxs = {
        idx for idx, row in enumerate(candidate_grid.rows) if _candidate_row_multiset(row)
    }
    native_numeric_idxs = {idx for idx, native_row in enumerate(native_rows) if native_row.multiset}
    result.row_binding_unverifiable = bool(
        candidate_numeric_idxs - bound_candidate_idxs or native_numeric_idxs - bound_native_idxs
    )

    # GH-273: numeric content and order establish row identity; only then may
    # the candidate stub verify that binding. Labels never choose a row (they
    # collide legitimately across panels), but a shifted, dropped, or invented
    # label is still a structural contradiction once the row is anchored.
    # Keep raw presence load-bearing alongside ``normalize_label``: that
    # normalizer deliberately erases presentation and punctuation, so a
    # punctuation-only non-empty label must not collapse into an empty stub.
    for cand_idx, native_idx in row_binding.items():
        result.row_labels_checked += 1
        candidate_label = candidate_grid.rows[cand_idx][0].strip()
        native_row = native_rows[native_idx]
        native_label = native_row.row_path[-1].strip() if native_row.row_path else ""
        same_presence = bool(candidate_label) == bool(native_label)
        candidate_key = normalize_label(candidate_label)
        native_key = normalize_label(native_label)
        if candidate_label and native_label and (not candidate_key or not native_key):
            # The shared row-label normalizer intentionally handles prose
            # labels, presentation, and footnotes; it does not canonicalize
            # mathematical notation (for example native ``β`` versus model
            # ``$\\beta$``). A non-empty label that normalizes to no key is
            # therefore not evidence of a mismatch. Fail closed as
            # unverifiable rather than falsely convicting or silently passing.
            result.row_label_unverifiable = True
        elif not same_presence or candidate_key != native_key:
            result.row_label_contradictions.append(
                RowLabelContradiction(
                    row_path=native_row.row_path,
                    candidate_label=candidate_label,
                    native_bbox=native_row.label_bbox,
                )
            )

    # BLOCKING 1: a native NUMERIC row that no candidate row ever bound to is not
    # merely "unverifiable" in the abstract — it is C4's dropped-digit signal
    # for the whole row. `_bind_rows` only tracks candidate coverage; a
    # dropped native row can still leave every candidate row bound (e.g. the
    # anchors either side of the gap still line up), so it must be checked
    # separately here rather than inferred from `len(row_binding)`.
    bound_native_idxs = set(row_binding.values())
    unbound_native_rows = [
        (idx, nr)
        for idx, nr in enumerate(native_rows)
        if nr.multiset and idx not in bound_native_idxs
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
    for cand_idx, cand_row in enumerate(candidate_grid.rows):
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

    # GH-352: the reported flag must describe the walk that actually RAN.
    #
    # It used the PROJECTED column count while the walk below gates on the
    # PHYSICAL one. `_project_candidate_data_columns` can make those disagree,
    # and then the 1:1 walk is skipped while the scoreboard reports columns
    # verified -- the 1/13 in the GH-332 table. `fully_checked` may still be 0
    # via `ambiguous_count` today, so it does not stamp SUCCESS yet; GH-326's
    # gate will read it, and then it would.
    #
    # Tied to the physical condition, which is the conservative direction: a
    # table can now only be called column-verifiable when the walk that would
    # verify it was actually performed.
    walk_column_binding_unverifiable = lane_count == 0 or physical_n_cand_cols != lane_count
    result.column_binding_unverifiable = walk_column_binding_unverifiable
    if walk_column_binding_unverifiable:
        # HIGH 1: column geometry itself is unverifiable, but that is not
        # licence to drop every cell signal for the whole table — only to
        # stop CLAIMING a binding for it. `_best_lane_column_map` salvages
        # what row_binding already proves: for each row we DO know binds,
        # any lane/column the map could not place anywhere (under ANY
        # admissible assignment) is unbound content, not an unknown.
        lane_to_col = _best_lane_column_map(
            native_rows,
            grid.rows,
            row_binding,
            lane_count,
            n_cand_cols,
        )
        lane_to_col = {
            lane: candidate_data_columns[col_idx] - 1
            for lane, col_idx in lane_to_col.items()
            if col_idx < len(candidate_data_columns)
        }
        mapped_lanes = set(lane_to_col.keys())
        mapped_cols = set(lane_to_col.values())
        col_to_lane = {col: lane for lane, col in lane_to_col.items()}

        for cand_idx, native_idx in row_binding.items():
            native_row = native_rows[native_idx]
            cand_row = candidate_grid.rows[cand_idx]
            if native_row.is_parent:
                _record_inventions_on_parent_row(
                    result,
                    native_row,
                    cand_row,
                    physical_n_cand_cols,
                    header_paths_by_lane,
                    col_to_lane,
                )
                continue

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

            for col_idx in range(physical_n_cand_cols):
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

    identity_col_to_lane = {i: i for i in range(n_cand_cols)}
    for cand_idx, cand_row in enumerate(grid.rows):
        if cand_idx not in row_binding:
            continue
        native_idx = row_binding[cand_idx]
        native_row = native_rows[native_idx]
        if native_row.is_parent:
            _record_inventions_on_parent_row(
                result,
                native_row,
                cand_row,
                n_cand_cols,
                header_paths_by_lane,
                identity_col_to_lane,
            )
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
                            native_bbox=native_row.lane_bboxes.get(lane),
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
                            native_bbox=native_row.lane_bboxes.get(lane),
                        )
                    )

    return result
