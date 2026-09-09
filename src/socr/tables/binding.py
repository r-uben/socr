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

import html
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum

from socr.tables.native_verifier import (
    _WELL_SEPARATED_GAP_PT,
    _cluster_x_positions,
    _lane_count_from_words,
    _normalize_numeric_token,
    _well_separated_lanes_in_row,
    is_numeric_token,
    label_key,
    label_key_is_bare_symbolic,
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
    #: #601: 0-indexed positions in ``rows`` whose label AND numeric
    #: multiset were both empty -- layout the model emitted between printed
    #: blocks (a blank gap), never a data row (a label-only row and a
    #: values-only row are both kept; see ``_is_spacer_row``).
    #:
    #: These rows are NOT removed from ``rows``: ``rows`` is indexed by the
    #: PHYSICAL 1-indexed row number the judge names in a cell ref
    #: (``table_verdict.resolve_cell_refs`` does exactly
    #: ``grid.rows[ref.row - 1]``), so dropping a row here would silently
    #: shift every later ref onto the wrong physical row. ``bind()`` is the
    #: one caller that filters spacer rows out, on its own internal working
    #: copy, never on the ``Grid`` it returns to other callers.
    spacer_row_indices: frozenset[int] = field(default_factory=frozenset)

    @property
    def spacer_rows_dropped(self) -> int:
        return len(self.spacer_row_indices)

    @property
    def n_cols(self) -> int:
        return (
            len(self.header_rows[0])
            if self.header_rows
            else (len(self.rows[0]) if self.rows else 0)
        )


def _split_row(line: str) -> tuple[str, ...]:
    return tuple(c.strip() for c in line.strip().strip("|").split("|"))


# --------------------------------------------------------------------------
# #601 / #624 -- candidate-row normalisation before bind()
#
# Owner rulings, 2026-09-08 (issues #601, #624, dispatched together):
#   #601: a candidate row with an empty label AND an empty numeric multiset
#     is layout (a printed blank gap), not data -- IDENTIFY it here, in the
#     candidate parser, but do not remove it from the returned ``Grid``:
#     ``table_verdict.resolve_cell_refs`` indexes that same ``Grid`` by the
#     PHYSICAL 1-indexed row a judge cell ref names, so dropping a row here
#     would silently shift every later ref onto the wrong row. ``bind()``
#     filters spacer rows out of its own internal working copy instead (see
#     ``Grid.spacer_row_indices``), and counts the drop for the audit trail
#     rather than silently changing anyone else's row count.
#   #624a: decode HTML entities and strip leading whitespace runs (including
#     U+00A0) from label cells at parse time -- the model sometimes encodes
#     sub-row indentation as literal ``&nbsp;`` entities in the label cell.
#     This IS baked into the returned ``Grid`` (same row count/order, so no
#     physical coordinate moves) -- every reader of this shared ``Grid``
#     (the binder AND ``table_verdict.resolve_cell_refs``) compares the real
#     label, not its markup.
#   #624b: a label-only row (non-empty label, every other cell empty)
#     immediately followed by a data row is a wrapped label -- merge its
#     text onto the next row's label with a single space, UNLESS the row is
#     a group header. A group header is identified POSITIVELY: its label
#     ends with ``--`` or ``:``, or it has >= 2 immediately-following child
#     rows that each carry a value in every numeric column while the header
#     row itself carries none, and a later sibling header exists. Never
#     inferred from "not obviously a data row" -- an unproven header must
#     merge like any other wrapped label (#601's empty spacer row still
#     drops, never merges).
# --------------------------------------------------------------------------

_LEADING_WS_RE = re.compile(r"^[\s ]+")


def _normalize_label_cell(text: str) -> str:
    """Decode HTML entities and strip leading whitespace (incl. U+00A0)."""
    return _LEADING_WS_RE.sub("", html.unescape(text))


def _is_spacer_row(row: tuple[str, ...]) -> bool:
    """#601: empty label AND empty numeric multiset -- layout, not data."""
    if not row:
        return True
    return not row[0].strip() and not _candidate_row_multiset(row)


def _is_label_only_row(row: tuple[str, ...]) -> bool:
    """Non-empty label, every other cell empty."""
    if not row or not row[0].strip():
        return False
    return all(not cell.strip() for cell in row[1:])


def _is_group_header_row(rows: tuple[tuple[str, ...], ...], i: int) -> bool:
    """Positive-only test: is row *i* (already known label-only) a header?

    Punctuation is the primary signal (``'Bank for International
    Settlements--'`` in the ruling's control). The structural fallback
    requires >= 2 immediately-following child rows that ALL carry a value in
    every numeric column, plus a later sibling label-only row -- a single
    following data row is exactly the wrapped-label shape (#624b's
    'Other authorized' / 'European currencies' control) and must merge, not
    stay a header.
    """
    label = rows[i][0].strip()
    if label.endswith("--") or label.endswith(":"):
        return True
    j = i + 1
    children: list[tuple[str, ...]] = []
    while j < len(rows) and not _is_spacer_row(rows[j]) and not _is_label_only_row(rows[j]):
        children.append(rows[j])
        j += 1
    if len(children) >= 2 and all(all(cell.strip() for cell in r[1:]) for r in children):
        if any(_is_label_only_row(rows[k]) for k in range(j, len(rows))):
            return True
    return False


def _normalize_candidate_rows(
    raw_rows: tuple[tuple[str, ...], ...],
) -> tuple[tuple[tuple[str, ...], ...], frozenset[int]]:
    """#624a label normalisation (applied to the returned rows) plus #601
    spacer identification (recorded, NOT applied).

    #624a is a pure cell-TEXT rewrite -- same row count, same row order --
    so it is safe to bake into the ``Grid`` every caller reads, physical-row
    consumers included: the shipped label is the plain one everywhere,
    without touching any physical coordinate.

    #601's spacer rows are only IDENTIFIED here (label cells are normalised
    first so entity/whitespace noise cannot hide a spacer row); removing
    them is left to ``bind()``'s own internal copy -- see
    ``Grid.spacer_row_indices``. #624b's wrapped-label merge is likewise
    deliberately NOT done here -- see ``_wrapped_label_merge_plan`` in
    ``bind()`` for why a text-only merge is unsafe and what native evidence
    it requires instead.
    """
    labeled = tuple(
        ((_normalize_label_cell(row[0]),) + row[1:] if row else row) for row in raw_rows
    )
    spacer_indices = frozenset(i for i, row in enumerate(labeled) if _is_spacer_row(row))
    return labeled, spacer_indices


def _wrapped_label_merge_plan(
    native_rows: list, rows: tuple[tuple[str, ...], ...]
) -> tuple[int, ...]:
    """#624b: candidate row indices that are a wrapped label merging onto
    ``rows[i + 1]``, proven against native geometry.

    A text-only rule is unsafe: this same label-only-row-followed-by-
    data-row SHAPE is also how a legitimate value-less parent row (a
    section/panel heading) or a units/footnote annotation sits above its
    first data row (see ``test_invented_digits_on_parent_heading_row_are_model_unbound``,
    ``test_candidate_valueless_units_row_absorbed_in_header_preserves_numeric_row_binding``)
    -- nothing in the candidate's own cell text tells those apart from a
    genuinely wrapped label with no lexicon or source-geometry access. This
    is also why the merge cannot live in ``parse_grid`` (markdown text only,
    no native words) -- it must run in ``bind()``, once native geometry
    exists.

    The native page does tell them apart. Run the ordinary anchor/
    interpolation binding once on the UNMERGED rows: a legitimate parent or
    units row binds to its own native counterpart (or is reported unbound
    with the next row's OWN label already matching native exactly -- nothing
    to gain by merging). A genuinely wrapped label's neighbour instead binds
    to a native row whose real label is LONGER than the neighbour's own
    label alone -- the source line the candidate split in two. Only THAT
    proven case merges: the merge is accepted only when the merged text is
    an exact ``label_key`` match for the native row it explains, never
    merely because the neighbour's own label came up short.

    GH-624b: that single-baseline proof (``row_path[-1]``) cannot see the
    case where NATIVE ITSELF prints the label across two baselines (a
    section heading is represented identically -- see the module's
    ``gh 624b parked`` history). Widening the proof to the native row's
    FULL ``row_path`` joined would catch that case, but is isomorphic to a
    legitimate parent-heading-row-plus-child and merges the heading too
    (``test_bound_parent_row_increments_row_labels_checked``). The owner
    ruled ``bind()`` may consume font evidence as the missing
    discriminator: a wrapped label's two printed lines are the SAME run of
    text (same face/size/weight); a heading is typographically distinct
    from its child. So the widened match only fires when BOTH native rows
    it explains -- the label-only row's own native band, and the
    following row's own native band -- carry equal, unambiguous
    ``label_font`` signatures (from the optional ``spans`` argument to
    ``bind()``). No spans, or disagreeing/ambiguous fonts, means no
    widened merge -- fail closed, matching every guard fixture, which
    supplies no font data at all.
    """
    if not rows:
        return ()

    trial_binding = _bind_rows(native_rows, rows)
    merge_at: list[int] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        if (
            _is_label_only_row(row)
            and i + 1 < len(rows)
            and not _is_label_only_row(rows[i + 1])
            and not _is_group_header_row(rows, i)
        ):
            next_row = rows[i + 1]
            native_idx = trial_binding.get(i + 1)
            native_label = (
                native_rows[native_idx].row_path[-1].strip()
                if native_idx is not None and native_rows[native_idx].row_path
                else ""
            )
            merged_label = f"{row[0].strip()} {next_row[0].strip()}".strip()
            proven = bool(native_label) and label_key(merged_label) == label_key(native_label)
            next_alone_already_matches = bool(native_label) and label_key(
                next_row[0].strip()
            ) == label_key(native_label)

            # GH-624b widened proof: native itself split the label across two
            # baselines. ``next_alone_already_matches`` is True in BOTH the
            # wrapped-label case and the heading case (the parked issue
            # comment measured this: the two fixtures are isomorphic on every
            # text/geometry signal), so it cannot gate the widened branch --
            # font evidence is the only discriminator, never text sameness.
            font_widened = False
            native_idx_this = trial_binding.get(i)
            if (
                native_idx is not None
                and native_idx_this is not None
                and native_rows[native_idx_this].is_parent
                and native_rows[native_idx].row_path
            ):
                native_label_wide = " ".join(
                    p.strip() for p in native_rows[native_idx].row_path
                ).strip()
                wide_proven = bool(native_label_wide) and label_key(merged_label) == label_key(
                    native_label_wide
                )
                if wide_proven:
                    font_a = native_rows[native_idx_this].label_font
                    font_b = native_rows[native_idx].label_font
                    font_widened = font_a is not None and font_a == font_b

            if (proven and not next_alone_already_matches) or font_widened:
                merge_at.append(i)
                i += 2
                continue
        i += 1

    return tuple(merge_at)


def _apply_wrapped_label_merges(
    rows: tuple[tuple[str, ...], ...], merge_at: tuple[int, ...]
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Collapse each ``rows[i]`` named in *merge_at* onto ``rows[i + 1]``.

    *merge_at* comes from :func:`_wrapped_label_merge_plan` run against the
    full-column candidate rows; applying the same index plan to a
    column-projected parallel row tuple (``grid.rows`` after
    ``_project_candidate_data_columns``) keeps both grids row-aligned with
    ``row_binding``, since column 0 (the label column) survives projection
    unchanged.
    """
    merge_set = set(merge_at)
    merged: list[tuple[str, ...]] = []
    merge_events: list[str] = []
    i = 0
    while i < len(rows):
        if i in merge_set:
            next_row = rows[i + 1]
            merged_label = f"{rows[i][0].strip()} {next_row[0].strip()}".strip()
            merged.append((merged_label,) + next_row[1:])
            merge_events.append(merged_label)
            i += 2
            continue
        merged.append(rows[i])
        i += 1

    return tuple(merged), tuple(merge_events)


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

        norm_rows, spacer_indices = _normalize_candidate_rows(tuple(body_rows))
        if len(spacer_indices) == len(norm_rows):
            continue  # every body row was a spacer -- no real table here

        return Grid(
            header_rows=tuple(header_block),
            rows=norm_rows,
            spacer_row_indices=spacer_indices,
        )

    return None


def _project_candidate_data_columns(grid: Grid) -> Grid:
    """Keep candidate columns that contain a genuine numeric data token.

    The rowizer emits a fixed lane grid before ``_clean_grid`` sees it.  A
    lane can consequently remain in the markdown when its header is populated
    but every body cell is empty; spec-number decorations such as ``(1)`` have
    the same header-only shape.  Such a column has no numeric binding claim.
    Projecting it out keeps the binder's column space aligned with the
    candidate's numeric data space without selecting a native lane by value.
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
        return grid
    widest_rows = max(len(columns) for columns in numeric_columns_by_row)
    data_columns = sorted(
        set().union(*(columns for columns in numeric_columns_by_row if len(columns) == widest_rows))
    )
    if not data_columns:
        return grid
    keep = (0, *data_columns)
    return Grid(
        header_rows=tuple(tuple(row[column] for column in keep) for row in grid.header_rows),
        rows=tuple(tuple(row[column] for column in keep) for row in grid.rows),
    )


def _candidate_data_column_indices(grid: Grid) -> tuple[int, ...]:
    """Return original indexes for the projected candidate data columns."""
    numeric_columns_by_row: list[set[int]] = []
    for row in grid.rows:
        columns = set()
        for column, cell in enumerate(row[1:], start=1):
            if any(
                is_numeric_token(token) and not _SPEC_NUMBER_RE.match(strip_presentation(token))
                for token in re.split(r"\s+", cell.strip())
                if token
            ):
                columns.add(column)
        numeric_columns_by_row.append(columns)
    if not numeric_columns_by_row:
        return tuple()
    widest_rows = max(len(columns) for columns in numeric_columns_by_row)
    return tuple(
        sorted(
            set().union(
                *(columns for columns in numeric_columns_by_row if len(columns) == widest_rows)
            )
        )
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
    # GH-624b: (basefont, rounded size, bold) for this row's label text, from
    # the optional ``spans`` argument to ``bind()``. None when no spans were
    # supplied, or when the label's own spans disagree on font -- either way
    # font evidence abstains rather than guesses (see ``_label_font_signature``).
    label_font: tuple[str, int, bool] | None = None


# PyMuPDF span ``flags`` bit for bold (see ``get_text("dict")`` docs): bit 4.
_SPAN_FLAGS_BOLD_BIT = 2**4


def _span_is_bold(span: dict) -> bool:
    """Bold via the documented PyMuPDF signals only -- no heuristic of our own."""
    flags = span.get("flags") or 0
    basefont = (span.get("font") or "").lower()
    return bool(flags & _SPAN_FLAGS_BOLD_BIT) or "bold" in basefont


def _bbox_overlaps(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    """True when two ``(x0, y0, x1, y1)`` boxes overlap in both x and y (strict)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return min(ax1, bx1) > max(ax0, bx0) and min(ay1, by1) > max(ay0, by0)


def _label_font_signature(
    spans: list[dict], label_bbox: tuple[float, float, float, float] | None
) -> tuple[str, int, bool] | None:
    """(basefont, rounded size, bold) for the spans overlapping *label_bbox*.

    None on no spans, no overlap, or disagreement among the overlapping spans
    -- ambiguous span evidence abstains rather than guesses, same as a missing
    ``spans`` argument (GH-624b design constraint 2).
    """
    if not spans or label_bbox is None:
        return None
    signatures: set[tuple[str, int, bool]] = set()
    for span in spans:
        bbox = span.get("bbox")
        text = (span.get("text") or "").strip()
        if not bbox or not text:
            continue
        if not _bbox_overlaps(tuple(bbox), label_bbox):
            continue
        signatures.add(
            (span.get("font") or "", round(span.get("size") or 0.0), _span_is_bold(span))
        )
    if len(signatures) != 1:
        return None
    return next(iter(signatures))


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


def _boxes_vertically_overlap(left: tuple, right: tuple) -> bool:
    """True when two ``(x0, y0, x1, y1, ...)`` word boxes overlap in y.

    Strict: boxes that merely touch are not overlapping. Same predicate the
    numeric-marker fold below already uses on extracted boxes; no distance
    tolerance is introduced.
    """
    return min(left[3], right[3]) > max(left[1], right[1])


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

    Numeric-free groups are never folded into other numeric-free groups.
    On the measured fixture (doc04 p3 ``1t`` under ``ROTATED PCs``) the
    subscript is a different ``(block_no, line_no)`` from its parent, its
    box top sits inside the parent height (also true of a short overlapping
    annotation such as ``(a)``), and the page's shorter-glyph height class
    mixes the ``1t`` with an on-line ``∗``, so no page-derived test
    separates a subscript from an annotation. The fold abstains rather
    than guess.
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
                    _boxes_vertically_overlap(word, numeric_word)
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
    words: list, n_cand_cols: int | None = None, spans: list[dict] | None = None
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
                    label_font=_label_font_signature(spans, label_bbox),
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
                label_font=_label_font_signature(spans, label_bbox),
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
            labels_match = bool(candidate_label and native_label) and label_key(
                candidate_label
            ) == label_key(native_label)
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
    #: #601: candidate body rows ``parse_grid`` dropped before binding
    #: because both the label and the numeric multiset were empty (layout,
    #: not data). 0 when ``parse_grid`` failed or nothing was dropped.
    candidate_spacer_rows_dropped: int = 0
    #: #624b: merged label text for each wrapped label-only row
    #: ``parse_grid`` joined onto the data row below it, in row order.
    candidate_wrapped_label_merges: tuple[str, ...] = ()
    #: Native rows and candidate→native map this call computed. Empty when
    #: parse_grid failed. Replay requires per-disputed-row evidence from these.
    native_rows: list = field(default_factory=list)
    row_binding: dict = field(default_factory=dict)
    candidate_row_labels: tuple[str, ...] = ()
    row_label_unverifiable_paths: tuple[tuple[str, ...], ...] = ()
    #: Words the region membership predicate rejected but that still touch
    #: *region* (positive overlap, at or under the majority-overlap
    #: threshold) — GH-609. A word with zero overlap carries no signal about
    #: the predicate's edge and is left out. Existing to make an excluded
    #: word visible instead of silently vanishing from the binding.
    boundary_words: list = field(default_factory=list)
    #: Subset of ``boundary_words`` that could still be table content --
    #: numeric-bearing, or clipped on both axes rather than a confidently
    #: external single-axis graze (GH-609 round 2, Astra P1). A non-empty
    #: list here means the binding is NOT ``fully_checked``: a possibly
    #: dropped cell must abstain rather than silently pass.
    unresolved_boundary_words: list = field(default_factory=list)
    #: The region-admitted words this attempt actually fed into row/column
    #: binding -- populated ONLY once ``bind()`` gets past the markdown parse
    #: gate (GH-609 round 5). Empty on a parse failure / no-numeric-lanes
    #: absence-of-evidence result, even though ``boundary_words`` /
    #: ``unresolved_boundary_words`` are computed earlier and so are NOT
    #: empty in that case. A caller that needs "was this specific word
    #: actually bound, on an attempt that really evaluated geometry" must
    #: read this field, not merely the absence of an unresolved entry.
    region_scoped_words: tuple = ()

    @property
    def fully_checked(self) -> bool:
        """True when nothing about this binding was left unresolved: every
        lane bound to exactly one candidate column, every native data row
        bound to exactly one candidate row, every bound row label checked,
        no cell's geometry was ambiguous, and no boundary-rejected word could
        still be table content. False means some region of the table was
        never actually compared — a different fact from whether the parts
        that WERE compared agreed (MAJOR 2)."""
        return (
            not self.row_binding_unverifiable
            and not self.row_label_unverifiable
            and not self.column_binding_unverifiable
            and self.ambiguous_count == 0
            and not self.unresolved_boundary_words
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


class BindingEvidence(str, Enum):
    """What one binding attempt actually establishes, as three closed values.

    P1 (owner rulings Q1/Q2). The gate's older helper was
    contradiction-only: it returned a ``BindingResult`` when something
    disagreed and ``None`` otherwise, which collapsed "structurally proven
    correct" and "nothing was checkable" into the same falsy answer. The
    ruled guard chain has to tell those apart -- a PASS overrules a reader
    outright, an ABSTAIN falls through to the blind-cell adjudicator.

    * ``PASS`` -- rows AND columns fully checked and nothing disagreed
      (``BindingResult.structural_agreement``). Never inferred from a
      matching numeric multiset: matching numbers prove "not invented",
      never "correctly placed", which is exactly the GH-273 shape.
    * ``CONTRADICT`` -- something that WAS checked disagreed: a contradicted
      cell, a row-label contradiction, or an unbound native/model cell (the
      dropped/invented-digit signal).
    * ``ABSTAIN`` -- no box, no native words, ``bind()`` failed, or coverage
      gaps left the question open. "We do not know", not "it matched".
    """

    PASS = "pass"
    CONTRADICT = "contradict"
    ABSTAIN = "abstain"


def classify_binding_evidence(result: BindingResult) -> BindingEvidence:
    """Classify one ``BindingResult`` into the three-way evidence vocabulary.

    Contradiction is asked FIRST: a table that disagreed somewhere is
    contradicted even if other regions were fully checked and agreed.
    """
    if not result.no_known_contradiction:
        return BindingEvidence.CONTRADICT
    if result.structural_agreement:
        return BindingEvidence.PASS
    return BindingEvidence.ABSTAIN


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


def _word_overlap_fraction(word: tuple, region: tuple[float, float, float, float]) -> float:
    """Fraction of *word*'s own box area that overlaps *region*.

    Denominator is the word's own box area, not the region's, so a small
    stub fully swallowed by a big region always scores ~1.0 while a big
    caption that only dips an edge into the region scores near 0.0. A
    degenerate (zero-area) word box scores 0.0 rather than dividing by zero.
    """
    rx0, ry0, rx1, ry1 = region
    wx0, wy0, wx1, wy1 = word[0], word[1], word[2], word[3]
    word_area = max(0.0, wx1 - wx0) * max(0.0, wy1 - wy0)
    if word_area <= 0.0:
        return 0.0
    ix0, iy0 = max(wx0, rx0), max(wy0, ry0)
    ix1, iy1 = min(wx1, rx1), min(wy1, ry1)
    inter_area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    return inter_area / word_area


def _word_majority_overlaps_region(word: tuple, region: tuple[float, float, float, float]) -> bool:
    """True when more than half of *word*'s own box area overlaps *region*.

    GH-609 (VI-A2 round 2). Replaces the pure centroid point-test, which two
    boundary shapes fell through:

    A stub whose ``x0`` sits a fraction of a point outside the region's
    min-x still scores ~1.0 here, same as it did under centroid (GH-331 /
    VI-A2) — one-sided sub-point overflow moves the area fraction and the
    centroid together, they never disagree on that shape.

    A caption or title WIDER than the table that also dips deep enough into
    the region from above (more than half its own height) used to slip
    through: its centroid can land inside the region even though the
    majority of its OWN box area sits outside it (bilateral x-overflow past
    both table edges pulls the box's total area down without moving its
    centroid, which tracks only the box's true midpoint). Majority-overlap
    area catches this; a point test on the midpoint cannot (cubic P2 — the
    "deeper overlap" case the shallow-graze fixture never exercised).

    A word whose top-left sits inside the region but whose box crosses out
    through the far edge by MORE than half its own extent on that single
    axis, with the other axis fully contained, is excluded by both tests
    identically: on one axis alone, "more than half the box overlaps" and
    "the box's midpoint is inside" are the same condition (both reduce to
    comparing the overlap length to half the word's own length), so this
    shape never lets majority-overlap-area be looser than centroid.

    This is NOT a general equivalence across two axes, though (round-2
    review correction): the 2-D area fraction is the PRODUCT of the two
    per-axis fractions, so area > 0.5 forces BOTH per-axis fractions above
    0.5, which is strictly more than either alone needs to be > 0.5. A word
    can have both per-axis fractions comfortably above 0.5 (its midpoint on
    each axis is inside, so centroid says "in") while their product is
    below 0.5 (majority-overlap-area says "out") -- the wide/deep caption
    above is exactly that shape, and it is a real, not merely academic,
    divergence. Any word this predicate rejects while still touching the
    region is recorded in ``BindingResult.boundary_words`` (never silent)
    rather than admitted outright; a subset that a numeric token or
    both-axes-clipped shape marks as possibly-still-table-content is
    additionally recorded in ``unresolved_boundary_words`` and forces
    ``fully_checked`` False (GH-609 round 2).
    """
    return _word_overlap_fraction(word, region) > 0.5


def _word_axis_overlap_fractions(
    word: tuple, region: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Per-axis overlap fraction of *word*'s own extent against *region*.

    ``(frac_x, frac_y)``, each in ``[0, 1]``: the fraction of the word's OWN
    width/height that lies within the region's x/y span. A degenerate
    (zero-width or zero-height) word box scores 0.0 on that axis.
    """
    rx0, ry0, rx1, ry1 = region
    wx0, wy0, wx1, wy1 = word[0], word[1], word[2], word[3]
    wdx, wdy = wx1 - wx0, wy1 - wy0
    frac_x = max(0.0, min(wx1, rx1) - max(wx0, rx0)) / wdx if wdx > 0.0 else 0.0
    frac_y = max(0.0, min(wy1, ry1) - max(wy0, ry0)) / wdy if wdy > 0.0 else 0.0
    return frac_x, frac_y


def _boundary_word_is_unresolved(word: tuple, region: tuple[float, float, float, float]) -> bool:
    """True when a boundary-rejected word might still be table content.

    Astra round-2 review, P1: recording a rejection in ``boundary_words``
    alone does not make a dropped numeric cell visible to evidence
    classification -- it has to affect whether the binding counts as fully
    checked. Two conditions mark a rejection as unresolved rather than
    confidently-external prose:

    A numeric token is ALWAYS unresolved: a dropped digit is exactly the
    silent-content-loss shape this repo forbids, so geometry never gets to
    wave a number away as prose.

    A non-numeric word is unresolved when it is clipped on BOTH axes (each
    per-axis overlap fraction is at least half) rather than fully contained
    on one axis and merely grazing the other. A pure single-axis graze --
    fully inside the region's x-range but only its top or bottom edge dips
    in (a caption from above, a footnote from below) -- has its entire
    departure from the region attributable to ONE direction; geometry alone
    can call that confidently external. A word partially clipped on both
    axes (the wide/deep caption; a label or overflowing cell spanning past
    two edges at once) cannot be told apart from a genuine table cell by
    geometry alone, so it stays open -- as does the exact half-and-half
    edge (old closed-interval centroid admitted it; strict majority
    rejects it), since ``>= 0.5`` on both axes catches that boundary too.
    """
    text = str(word[4]).strip() if len(word) > 4 else ""
    if text and is_numeric_token(text):
        return True
    frac_x, frac_y = _word_axis_overlap_fractions(word, region)
    return frac_x >= 0.5 and frac_y >= 0.5


def _partition_words_by_region(words: list, region: tuple | None) -> tuple[list, list, list]:
    """Split *words* into (kept, boundary, unresolved_boundary).

    ``kept`` holds words admitted under ``_word_majority_overlaps_region``.
    ``boundary`` holds words the predicate rejected but that still touch
    *region* (positive overlap, at or under the half-area threshold) — GH-609:
    a word with zero overlap carries no signal about the predicate's edge and
    is left out of both lists. ``unresolved_boundary`` is the subset of
    ``boundary`` that ``_boundary_word_is_unresolved`` cannot rule out as
    table content (round 2). ``region=None`` returns *words* unchanged with
    both other lists empty — byte-for-byte the old unscoped behaviour.
    """
    if region is None:
        return words, [], []
    try:
        x0, y0, x1, y1 = (float(v) for v in region)
    except (TypeError, ValueError):
        return words, [], []  # a malformed region is an absence of scoping, not a conviction
    if not (x0 <= x1 and y0 <= y1):
        return words, [], []
    box = (x0, y0, x1, y1)
    kept: list = []
    boundary: list = []
    unresolved: list = []
    for w in words:
        frac = _word_overlap_fraction(w, box)
        if frac > 0.5:
            kept.append(w)
        elif frac > 0.0:
            boundary.append(w)
            if _boundary_word_is_unresolved(w, box):
                unresolved.append(w)
    return kept, boundary, unresolved


def _words_in_region(words: list, region: tuple | None) -> list:
    """Filter *words* to those whose box majority-overlaps *region*.

    GH-330 / GH-609. ``bind`` was only ever called with a whole page's words, so
    on a page with prose above the table and notes below it, lane clustering ran
    over text that is not in the table at all — which is why column binding was
    unverifiable on every real page measured. Every native table region already
    arrives as a ``(rect, markdown)`` pair, so the rect was available all along
    and simply never passed in.

    Majority overlap area, not top-left and not a centroid point test: a stub
    whose ``x0`` is 10⁻³ pt left of the region's min-x is still a table word
    (GH-331 / VI-A2); a caption whose box dips a fraction of a point in from
    above is not; a caption WIDER than the table that dips deep into it from
    above is not, even though its centroid can land inside (GH-609). Kept
    here rather than imported so ``binding`` stays free of ``fitz``.

    ``region=None`` returns *words* unchanged — byte-for-byte the old behaviour.
    Callers that need the rejected-but-touching words too should call
    ``_partition_words_by_region`` directly instead.
    """
    kept, _boundary, _unresolved = _partition_words_by_region(words, region)
    return kept


def bind(
    words: list,
    markdown: str,
    *,
    region: tuple | None = None,
    spans: list[dict] | None = None,
) -> BindingResult:
    """Bind *markdown*'s candidate grid to the native geometry in *words*.

    *spans*, when given, is a flat list of PyMuPDF span dicts (the ``spans``
    entries of ``page.get_text("dict")``'s blocks/lines, each carrying
    ``bbox``, ``size``, ``flags`` and ``font``/basefont). It is OPTIONAL and
    additive: with ``spans=None`` (the default), behaviour is byte-for-byte
    what it was before GH-624b, including the ambiguous wrapped-label-vs-
    heading case, which stays unmerged. See ``_wrapped_label_merge_plan``.

    Never raises on malformed input: a markdown block that fails the A1
    strict parse, or a page with no numeric lanes, binds nothing and returns
    an (empty) :class:`BindingResult` — an absence of evidence, not a
    conviction of either side.

    *region*, when given, is the candidate's own ``(x0, y0, x1, y1)`` extent; words
    outside it are dropped before any geometry is computed (GH-330). Omitting it is
    the unscoped whole-page fallback, whose column binding is expected to be
    unverifiable on any page that carries text outside the table. A word the
    region predicate rejected but that still touched *region* is recorded on
    the result's ``boundary_words`` (GH-609) rather than vanishing silently;
    the subset that might still be table content additionally lands in
    ``unresolved_boundary_words`` and forces ``fully_checked`` False
    (round 2).
    """
    words, boundary_words, unresolved_boundary_words = _partition_words_by_region(words, region)
    result = BindingResult()
    result.boundary_words = boundary_words
    result.unresolved_boundary_words = unresolved_boundary_words

    grid = parse_grid(markdown)
    if grid is None:
        return result
    result.candidate_spacer_rows_dropped = grid.spacer_rows_dropped

    # #601: filter spacer rows out of bind()'s OWN working copy only. The
    # ``Grid`` parse_grid returned keeps every physical row (see
    # ``Grid.spacer_row_indices``'s docstring) -- other callers, chiefly
    # ``table_verdict.resolve_cell_refs``, index it by the physical
    # 1-indexed row a judge cell ref names, and must never see a row count
    # bind() has quietly changed.
    if grid.spacer_row_indices:
        grid = replace(
            grid,
            rows=tuple(row for i, row in enumerate(grid.rows) if i not in grid.spacer_row_indices),
        )

    # GH-609 round 5: the region-admitted words this attempt ACTUALLY fed into
    # row/column binding, set only past the parse-failure gate above. This is
    # the "was this word really evaluated" signal a caller needs to tell a
    # positive resolution ("bound, on a candidate that parsed") apart from an
    # absence of evidence ("bind() never got past parse_grid") -- an empty
    # ``unresolved_boundary_words`` on the latter proves nothing about any
    # specific word, admitted or not.
    result.region_scoped_words = tuple(words)

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

    native_rows, band_centers, header_band_idxs = _native_rows(words, n_cand_cols, spans)
    lane_count, _lane_of, lane_centers = _native_lane_geometry(words, n_cand_cols)

    header_words = (
        _native_header_words(words, band_centers, header_band_idxs) if header_band_idxs else []
    )
    result.column_header_paths = build_column_header_paths(words, grid, lane_centers, header_words)

    # #624b: merge a wrapped label-only row onto the data row below it, once
    # native geometry exists to prove the merge (see
    # ``_wrapped_label_merge_plan`` for why this cannot run in parse_grid).
    # The same index plan is applied to both ``candidate_grid`` (full
    # columns, what the plan was computed against) and ``grid`` (possibly
    # column-projected) so every downstream ``cand_idx`` -- row_binding,
    # candidate_row_labels, the cell walks below -- stays aligned across
    # both.
    merge_at = _wrapped_label_merge_plan(native_rows, candidate_grid.rows)
    if merge_at:
        merged_candidate_rows, merge_events = _apply_wrapped_label_merges(
            candidate_grid.rows, merge_at
        )
        candidate_grid = replace(candidate_grid, rows=merged_candidate_rows)
        if grid is not None:
            merged_grid_rows, _ = _apply_wrapped_label_merges(grid.rows, merge_at)
            grid = replace(grid, rows=merged_grid_rows)
        result.candidate_wrapped_label_merges = merge_events

    # I1 BIDIRECTIONALITY: row-level binding and its unbound-row signals run
    # regardless of whether column geometry (lane_count vs n_cand_cols) is
    # even usable this call. `_bind_rows` anchors rows from whole-row
    # multisets, independent of lane geometry, so gating this behind the
    # lane/column check (an early `return` used to) silently swallowed
    # every row-level drop/invention signal whenever column geometry was
    # ALSO unverifiable — this is exactly what HIGH 1 and HIGH 2 reported.
    row_binding = _bind_rows(native_rows, candidate_grid.rows)
    result.native_rows = list(native_rows)
    result.row_binding = dict(row_binding)
    result.candidate_row_labels = tuple(
        (row[0].strip() if row else "") for row in candidate_grid.rows
    )
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
    unverifiable_paths: list[tuple[str, ...]] = []
    for cand_idx, native_idx in row_binding.items():
        result.row_labels_checked += 1
        candidate_label = candidate_grid.rows[cand_idx][0].strip()
        native_row = native_rows[native_idx]
        native_label = native_row.row_path[-1].strip() if native_row.row_path else ""
        same_presence = bool(candidate_label) == bool(native_label)
        candidate_key = label_key(candidate_label)
        native_key = label_key(native_label)
        candidate_unprovable = not candidate_key or label_key_is_bare_symbolic(candidate_key)
        native_unprovable = not native_key or label_key_is_bare_symbolic(native_key)
        if candidate_label and native_label and (candidate_unprovable or native_unprovable):
            # The shared row-label normalizer intentionally handles prose
            # labels, presentation, and footnotes; it does not canonicalize
            # mathematical notation (for example native ``β`` versus model
            # ``$\\beta$`` -- a bare symbolic label, GH-585's
            # ``label_key_is_bare_symbolic``). A non-empty label that keys
            # to nothing provable is therefore not evidence of a mismatch.
            # Fail closed as unverifiable rather than falsely convicting or
            # silently passing.
            result.row_label_unverifiable = True
            unverifiable_paths.append(tuple(native_row.row_path))
        elif not same_presence or candidate_key != native_key:
            result.row_label_contradictions.append(
                RowLabelContradiction(
                    row_path=native_row.row_path,
                    candidate_label=candidate_label,
                    native_bbox=native_row.label_bbox,
                )
            )
    result.row_label_unverifiable_paths = tuple(unverifiable_paths)

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
