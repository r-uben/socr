# 2026-08-13 — PR #192 review response (GH-144 A2/A2b)

Branch: `fix/144-a2-text-strategy-grid`. Fixed the four blocking findings from
the PR #192 review of `src/socr/tables/reconstruct.py`'s A2/A2b rejection
path (the text-strategy grid GH-144's word-geometry rowizer falls back from).

## Finding 1 — early return dropped other tables on the page

`reconstruct_table_regions`'s per-table loop did `return rowized` / `return
[]` on rejection, discarding every table already collected in `out` plus
every table still to come. Changed to `out.extend(rowized); continue` (and
plain `continue` when the fallback finds nothing), so one damaged table no
longer takes the rest of the page down with it.

Test: `tests/test_reconstruct_gh144_review.py::
test_multi_table_page_keeps_undamaged_tables_when_one_is_rejected` — a real
damaged table (from the `table_page` fixture) plus a fake clean table
monkeypatched into `page.find_tables`'s result; asserts both survive.

## Finding 2 — false rejects from an overrun `table.bbox`

`_destroyed_numeric_tokens` scoped its search to `table.bbox`, which
`find_tables` infers from whitespace and routinely overruns past the last
data row into unrelated text below (that overrun is how the notes paragraph
ended up inside the table region in the first place — see A1 log §3). A
numeral caught in that overrun with no containing cell would count as
"destroyed" and reject an otherwise-clean grid.

Fix: scope both the destruction check and the fallback rowizer to
`_numeric_row_bbox`'s tighter, data-driven union of the table's own
numeric-bearing rows, not `table.bbox`.

Test: `tests/test_reconstruct_gh144_review.py::
test_destroyed_check_is_scoped_to_numeric_rows_not_overrun_bbox` — a
mechanistic unit test with a fake table whose `bbox` overruns 20pt into a
stray "106" with no containing cell; asserts `table.bbox` scoping falsely
flags it (sanity) and `_numeric_row_bbox` scoping does not.

## Finding 3 — the numeric-row scope defeated A2b's own header-prepend

`_numeric_row_bbox`'s scope is, by construction, the union of rows carrying
a numeric cell — a header band carries none, so scoping the fallback's word
list to that alone silently excluded the header before
`rowize_from_word_list` ever ran. A2b's `_prepend_header_band` had nothing
left to prepend exactly when A2 fires against it.

Fix: `_extend_scope_for_header` walks upward from the numeric scope's top
edge, absorbing a preceding lane-snapping, non-numeric header band (mirrors
`_prepend_header_band`'s own absorption rule for consistency). Also fixed
two bugs found while verifying this end-to-end against the `table_page`
fixture:

- `_numeric_row_bbox` used `table.rows[n].bbox` alone, which silently
  excluded native word bboxes that fall outside the row's official rect
  (numeric row expanded to also union real word bboxes whose centre falls
  in the row).
- `_extend_scope_for_header`'s upward walk compared `round(w[1])` row keys
  against the raw float `tight.y0` — `round(94.325) == 94 < 94.325` treated
  the row that DEFINES the scope as "above" itself and broke the walk
  before it ever reached the real header. Fixed by rounding the boundary
  too.
- `_cell_has_numeric_token` normalizes raw `extract()` cells (splits on
  whitespace, requires every token to match `_NUM_TOKEN_RE`) instead of
  matching the unnormalized cell string directly — a cell like `"0.67\n"`
  or a merged multi-value cell was invisible to a bare regex match, which
  silently defeated the tight scope and (per the review's own warning)
  risked reopening the notes-merge regression via silent full-page
  broadening.

Test: `tests/test_region_overlap_gh145.py::
test_rejected_grid_fallback_populates_the_header` — asserts the fallback's
header row contains `Nominal`/`Real`/`Inflation`, not just that the numeric
values survive.

## Finding 4 — severity: rejection ships as a quiet WARNING

Surfacing "grid destroyed numeric tokens" at page/document/CLI level needs
`born_digital.py` / `orchestrator.py`, both out of this ticket's write set
(owned by GH-147 A2 / PR #193, open concurrently). Filed
[GH-195](https://github.com/r-uben/socr/issues/195) instead of implementing
it here, citing A1 log §3 and the house "no silent content loss" rule.
Referenced in the rejection-site warning via `_SILENT_LOSS_FOLLOWUP_ISSUE`.

## Nits applied

- Aggregated per-token warnings into one `logger.warning` per rejected
  table (was already the shape after the finding-1 rewrite).
- `_prepend_header_band`'s absorption loop now rejects a candidate header
  row that itself carries any numeric token, even if its words snap to the
  segment's lanes (a second table's trailing data row stacked directly
  above).
- `_prepend_header_band` is now only called when the previous segment was
  NOT already consumed (`consumed: set[int]` tracking in
  `rowize_from_word_list`).
- Removed `_rowize_segment`'s now-redundant internal padding/column-drop
  block — every row is `[label] + row_cells` with `row_cells` uniformly
  `len(lane_centers)` wide by construction, so `_prepend_header_band`'s own
  `lane_centers`-based row width already matches without a second, separate
  drop step.

## Nits skipped

- **Shared `is_numeric_token` predicate** (`native_verifier.py`): would
  touch 6+ call sites in `reconstruct.py` (some pre-existing, outside this
  ticket's four findings), each needing the same lazy-import guard
  `_is_data_row` already uses to avoid the circular import
  (`native_verifier` imports `_NUM_TOKEN_RE` from this module at module
  level), and changes matching semantics (presentation-stripping, currency
  prefixes, unicode minus) with no dedicated test coverage for those paths
  here. Better as its own reviewed change.
- **Snap on word centre/full box, not `x0` alone**: `_numeric_lane_centers`
  is documented as a byte-faithful mirror of `_rowize_segment`'s own
  x0-based lane-clustering, specifically so A2b's header-detection stays
  consistent with the rowizer's own placement. Changing `_snaps` to
  word-centre while `_numeric_lane_centers` stays x0-based would create a
  lane-centre/snap-test mismatch; making both consistent would mean editing
  the mirrored lane-placement logic itself — the surface A1's retarget
  explicitly prohibits touching.

## Verification

- `~/venvs/socr/bin/pytest tests/ -q` → 1575 passed, 1 xfailed, 0 failed
  (baseline was 1572 passed / 1 xfailed / 0 failed; +3 new tests).
- `uvx ruff@0.16.0 format --check .` → clean after `uvx ruff@0.16.0 format`
  reformatted the 2 new test files (multi-line assert/list wrapping only).
- Per-finding coverage: finding 1 →
  `test_multi_table_page_keeps_undamaged_tables_when_one_is_rejected`;
  finding 2 → `test_destroyed_check_is_scoped_to_numeric_rows_not_overrun_bbox`;
  finding 3 → `test_rejected_grid_fallback_populates_the_header`.

## Write set

`src/socr/tables/reconstruct.py`, `tests/test_region_overlap_gh145.py`,
`tests/test_reconstruct_gh144_review.py` (new). Did not touch
`src/socr/core/born_digital.py` or `src/socr/pipeline/orchestrator.py`
(GH-147 A2 / PR #193 territory).
