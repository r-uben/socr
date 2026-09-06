# 2026-09-06 — TICKET-A1b: structure-class selection consults row corroboration before the floor

Issue: #634. Depends on TICKET-A1a (#627, `src/socr/tables/row_corroboration.py`, merged to
`main` as PR #630). Branch: `feat/634-corroboration-selection`, cut from `main@5728847`.

## What changed

`src/socr/core/manifest.py`:

- `_row_corroborated_grid_winner(p)` — S1 case (i)-b: when `_strict_grid_authored_pool` is
  empty (the fail-closed floor would otherwise apply), scores every candidate the strict pool
  discarded with A1a's `corroborate_rows` against the page's detected table region. A
  candidate that clears (share/extra-share/skipped-rows gate), passes a row-count-allowance
  check, and passes an edge-row check becomes the winner. Tie-break: not-ladder-rejected,
  then corroboration share, then the existing `(audit_passed, confidence, word_count)`
  ordering.
- `_grid_reading_attempt(out)` — the candidate-pool predicate for the fallback. Uses
  `has_authored_table_grid` (ragged body allowed), not `has_strict_table_grid`
  (`_grid_shaped_attempt`'s own predicate). **Measured against the real ECB fixture cache,
  0/6 real qwen/gemini candidates pass the strict uniform-body-width requirement** — a
  spanning units/legend row above the numeric body is routine in real statistical tables —
  so the strict predicate was silently starving the fallback of any candidate to score at
  all. This is a real bug fix, not a widening of scope: without it the ticket's own six
  motivating fixtures never reach `corroborate_rows`.
- `REGION_COVERAGE_MIN_SHARE = 0.75` — named constant, see "Region-selection bug" below.
- `_row_count_allowance_ok(rc)` — candidate/native row-count reconciliation (closes the
  A1a→A2 truncation window: `clears` alone can pass a candidate that dropped whole rows off
  the end, since its denominator is the candidate's own row count).
- `_edge_row_skip_count(words, markdown, region)` — native numeric bands, within region,
  before the candidate's first bound row or after its last one (a class
  `RowCorroboration.skipped_native_rows` is blind to: that gate only counts gaps strictly
  between two bound rows).
- `_splice_unverified_row_markers` / `_apply_row_corroboration_disclosure` — per-row
  `<!-- row unverified -->` markers spliced into the shipped markdown for unbound rows, plus
  an `AuditEvent` (`structure_class_row_corroborated`) naming the header text as emitted, the
  bound/total counts, and (round 2, below) `corroboration_region` / `region_coverage_share`.
- `_derive_structure_floor_overrides` (`src/socr/core/audit_log.py`) — emits
  `structure_floor_overrode_ladder` when a ladder-ACCEPTED candidate is still floored,
  generalising #589's fix (option c).
- Both new audit-event kinds (`structure_class_row_corroborated`,
  `structure_floor_overrode_ladder`) registered in `audit_log.py`'s rank dict and
  `tables_trust.py`'s `TABLE_DISTRUST_KINDS`.
- `src/socr/tables/row_corroboration.py` — eight private helpers renamed to public
  (`_baseline_bands` → `baseline_bands`, `_is_column_index_row` → `is_column_index_row`,
  `_match_rows_monotonic` → `match_rows_monotonic`, `_numeric_body_rows` →
  `numeric_body_rows`, `_table_blocks` → `table_blocks`, `_words_in_region` →
  `words_in_region`, `_is_separator_row` → `is_separator_row`, `_split_cells` →
  `split_cells`). See "Layering-test fix" below — this file is outside this ticket's
  original write-ownership list; the rename is a disclosed, mechanical exception, forced by
  a test gate this ticket's own code tripped.

New tests `tests/test_s1_structure_class_winner_corroboration.py` — 6 hermetic tests
(synthetic `PageState`, no PDF/provider): strict-pool-empty precondition, corroborating
candidate wins over the floor, non-corroborating candidate alone still floors, no-native-words
is identical to the pre-A1b floor, and per-row marker placement (independent of the `clears`
gate's own threshold tuning).

## CONSILIUM-GATE: outcome, not flag

Ticket text named `native_table_header_unattributed` as the fallback's gate condition.
**Owner ruling (2026-09-06):** the census's `header_unattributed` defect actually came from a
MODEL-side audit event (`table_structure_failed: header_unattributed`, engine gemini/qwen),
not the native page flag — which measures `False` on all six of the ticket's own motivating
fixtures. Gating on the literal flag would make the fallback a no-op everywhere it was meant
to fire. Resolution (option 2): `_row_corroborated_grid_winner` fires iff the strict
grid-authored pool is empty (the floor would otherwise ship the fail-closed marker) — never
competes with a strict-pool winner, and C1's invariant (native never authors the grid) is
untouched, since what ships is a MODEL candidate the native layer corroborates.

## Region-selection bug (round 2, owner correction)

First pass scored every candidate against `region = union of detected_table_bboxes`, per the
ticket's literal spec. **Result: none of the six motivating fixtures' real candidates
cleared** — `corroborate_rows` measured 40-100% of each candidate's numbers as "extra". This
was reported as a negative finding and **retracted by the owner**: A1a's own log measured
extras 0.0-0.0077 on these SAME candidates with region = whole page. The defect was region
selection, not candidate quality.

### Measurement: (a) page / (b) per-bbox / (c) union numeric-word coverage

| Fixture | bboxes | (a) page | (b) per-bbox | (c) union |
|---|---|---|---|---|
| bulletin p1 | 1 | 517 | [16] | 16 |
| bulletin p2 | 1 | 441 | [18] | 18 |
| bulletin p3 | 1 | 410 | [16] | 16 |
| report p1 | 1 | 200 | [15] | 15 |
| report p2 | 2 | 502 | [18, 13] | 306 |
| report p3 | 2 | 606 | [17, 16] | 326 |

(counts are genuine numeric words, `_is_genuine_numeric`, inside each region)

### (d) `corroborate_rows` under region = bbox-union vs region = whole page

| Fixture | engine | bbox-union bound/total | extra_share | whole-page bound/total | extra_share |
|---|---|---|---|---|---|
| bulletin p1 | qwen | 0/39 | 1.0 | 39/39 | 0.0 |
| bulletin p2 | qwen | 0/39 | 1.0 | 36/39 | 0.0075 |
| bulletin p3 | gemini | 0/39 | 1.0 | 39/39 | 0.0 |
| report p1 | gemini | 0/14 | 1.0 | 14/14 | 0.0 |
| report p2 | gemini | 18/36 | 0.400 | 36/36 | 0.0 |
| report p3 | gemini | 18/36 | 0.484 | 36/36 | 0.0 |

The whole-page column reproduces A1a's own measurement exactly, confirming the bbox
was the defect, not the candidates.

### Fix: coverage-gated region widening

`REGION_COVERAGE_MIN_SHARE = 0.75`. Per candidate: score against the bbox union first
(`rc_bbox`); if `rc_bbox.native_numeric_rows / rc_bbox.total < REGION_COVERAGE_MIN_SHARE`,
the detector under-covers and the `clears` gate is re-scored against the whole page
(`region=None`, `words_in_region`'s own no-op sentinel). Measured coverage ratios that
justify 0.75: all six fixtures' bbox-union coverage was 0.10-0.69 (16/39, 18/39, 5/39, 4/14,
25/36, 23/36 — bands, not raw words), while the SAME six fixtures' whole-page coverage was
1.28-1.93 — a wide, unambiguous gap.

**Second-order defect found while implementing this, also fixed:** widening the row-count-
allowance and edge-row checks to the SAME whole-page region as `clears` breaks them —
`native_numeric_rows` on the whole page counts every unrelated numeric band on the page
(dates, footnote numbers, other tables), not just this table's rows (measured: 50 page-wide
bands vs. 39 real table rows on bulletin p1, failing the allowance check outright; 5 spurious
edge-row counts on report p2's two-block table, where half the true rows sit in a second
block entirely outside the narrow bbox). Fix: once the region has been judged unreliable
(coverage below threshold), the row-count-allowance and edge-row checks are skipped rather
than scored against either extreme — the bbox is wrong in one direction (near-empty single
blocks) or the other (a moderately-covering bbox from a multi-block table), and the page is
wrong for a different reason (counts unrelated content). A1a's own `clears` gate already
covers interior gaps (`skipped_native_rows`) at whatever region is actually scored; only the
outside-the-bound-range case goes unchecked in the widened branch — a disclosed narrowing of
this ticket's own edge-row addition, not of A1a's contract.

`corroboration_region` (`"bbox_union"` / `"page"`) and the pre-widen coverage share are
recorded on the `structure_class_row_corroborated` audit event's `data` field and in its
`detail` string, so a partial bbox that forced page-wide scoring is visible, never silent.

### Final winner table (post-fix)

| Fixture | winner | bound/total | share | region | coverage_share | unbound rows |
|---|---|---|---|---|---|---|
| bulletin p1 | qwen | 39/39 | 1.000 | page | 0.103 | none |
| bulletin p2 | qwen | 36/39 | 0.923 | page | 0.128 | rows 4, 12, 24 |
| bulletin p3 | gemini | 39/39 | 1.000 | page | 0.128 | none |
| report p1 | gemini | 14/14 | 1.000 | page | 0.286 | none |
| report p2 | gemini | 36/36 | 1.000 | page | 0.694 | none |
| report p3 | gemini | 36/36 | 1.000 | page | 0.639 | none |

Matches the census's own named winners (bulletin p1/p2 qwen, p3 gemini; report p1-p3
gemini), all now shipped flagged (per-row markers where any row is unbound) rather than
discarded to the fail-closed floor.

## Layering-test fix

`manifest.py`'s `_edge_row_skip_count` and `_table_block_layout` import
`row_corroboration`'s block/band helpers by underscore name across the `core`/`tables`
package boundary — a violation `tests/test_package_layering.py::test_no_private_symbol_imported_across_packages`
forbids (its allowlist carries an explicit "do not add entries" note, honored: nothing was
added to it). Fix: the eight helpers were genuinely public API already in all but name (used
only internally to `row_corroboration.py` plus these two cross-package call sites) — renamed
to drop the leading underscore in `row_corroboration.py`, and the two `manifest.py` imports
updated. No behavior change; confirmed via `grep` that no other file in the repo references
the old private names.

## Disclosed judgment calls

- The row-count-allowance and edge-row checks (this ticket's own additions on top of A1a)
  are skipped, not scored, once the region has been judged unreliable — see "Region-selection
  bug" above.
- `row_corroboration.py` is outside this ticket's original write-ownership list; the rename
  was necessary to keep `manifest.py`'s existing (pre-region-fix) private cross-package
  imports from failing a hermetic layering test the code already tripped before this session's
  region work began.
- Header stub text is recorded in the sidecar as emitted (no neutral `col N` rewrite) per the
  ticket's own "owner may rule later" note.

## What's left for A1c

A1c (a separate, later-wave ticket per `docs/plans/defect-census-fixes/TICKETS.md`, own files:
`result.py`, `cli.py`, orchestrator resume-gate) is untouched by this ticket. The
`structure_class_row_corroborated` audit event (with its `corroboration_region` /
`region_coverage_share` fields) is the intended attachment point for A1c's own marker work.

## Round 3: row-shape reconciliation replaces both distance-anchored checks

A reviewer repro on the row-count-allowance/edge-row checks above (round 2's own additions,
scored against the bbox-union region) found a real gap: deleting a single complete data row
from the real bulletin p1 qwen candidate (e.g. the `2018` row) still let `corroborate_rows`
report `clears=True` — `bound/total = 38/38 = 1.0 ≥ 36/39`, because the gate's denominator is
the candidate's OWN row count, so a dropped row lowers numerator and denominator together and
the allowance never fires. The round-1/2 checks that were meant to catch exactly this (the
row-count-allowance, the edge-row walk) were anchored to `detected_table_bboxes`, which round
2's own investigation had already shown does not cover the numeric body on any fixture — so
in practice those checks fired on almost nothing.

### Round 2 follow-up: band-run (distance-anchored) — measured, rejected

Before dropping distance entirely, a "band-run" alternative was measured: extend the checked
region from the bbox union by up to `N` row-pitches in either direction, to reach real data
rows without reaching unrelated content. Requested measurement — internal row pitch, the
largest within-table gap (multi-block tables), and the smallest table-to-non-table gap, both
raw and pitch-normalized:

| fixture | pitch | max within-table gap | int/pitch | gap→footnote | after/pitch | gap→boundary | before/pitch |
|---|---|---|---|---|---|---|---|
| bulletin p1 | 7.68 | 23.62 | 3.08 | 20.15 | 2.62 | 21.60 | 2.81 |
| bulletin p2 | 7.68 | 23.16 | 3.02 | 20.17 | 2.63 | 21.60 | 2.81 |
| bulletin p3 | 7.68 | 23.64 | 3.08 | 20.17 | 2.63 | 21.60 | 2.81 |
| report p1 | 7.92 | 11.76 | 1.48 | 28.45 | 3.59 | 11.50 | 1.45 |
| report p2 | 7.92 | 35.06 | 4.43 | 28.22 | 3.56 | 11.50 | 1.45 |
| report p3 | 7.68 | 48.43 | 6.31 | 42.10 | 5.48 | 11.50 | 1.50 |

Max within-table gap across the set: **6.31×** pitch (report p3's own internal two-block
break). Min table-to-non-table gap across the set: **1.45×** pitch (report p1/p2/p3's
boundary gap). The former is larger than the latter — no single raw or pitch-normalized
threshold admits every real multi-block continuation while excluding every real boundary.
Per the owner's own stop condition ("if they overlap on any fixture, report the numbers and
stop"), this was reported and the band-run design was dropped without being implemented.

### Design: row shape, no distance or geometry

Replaces `_row_count_allowance_ok` and `_edge_row_skip_count` (both removed) with a single
`_row_shape_reconciliation_ok(words, markdown)`:

- `ROW_SHAPE_MIN` — data-derived per candidate: the minimum numeric-token count over the
  candidate's own `numeric_body_rows`. No named constant; footnote bands (one bare number),
  date/label-only bands, and value-less section headings are narrower than any real data row
  and fall out on their own.
- `native_table_rows` — count of whole-page (`region=None`, matching what `corroborate_rows`
  already scores the `clears` gate against once under-coverage widens it) native baseline
  bands whose own numeric-token count is `>= ROW_SHAPE_MIN` and that are not the printed
  column-index legend row (`is_column_index_row`).
- Reconciliation: `candidate_total >= ceil(native_table_rows * ROW_CORROBORATION_MIN)` (A1a's
  own 36/39 allowance), run unconditionally, regardless of `region_kind`. A page with two
  independent tables where the candidate covers only one is rejected by design — the
  page-level `native_table_rows` counts both tables, so a candidate missing an entire second
  table cannot reconcile.

### Measurement (required before implementing, per owner instruction)

Real winning candidate vs whole-page `native_table_rows`, all six fixtures:

| fixture | engine | ROW_SHAPE_MIN | candidate_total | native_table_rows | ratio |
|---|---|---|---|---|---|
| bulletin p1 | qwen | 12 | 39 | 39 | 1.000 |
| bulletin p2 | qwen | 10 | 39 | 39 | 1.000 |
| bulletin p3 | gemini | 8 | 39 | 39 | 1.000 |
| report p1 | gemini | 12 | 14 | 14 | 1.000 |
| report p2 | gemini | 10 | 36 | 36 | 1.000 |
| report p3 | gemini | 15 | 36 | 36 | 1.000 |

Every real winner reconciles exactly (no strays, no under-count) — all complete candidates
pass, per the owner's own stop condition (only a failing complete candidate would have
blocked implementation).

Reviewer's repro, run directly against `_row_corroborated_grid_winner` with rows
programmatically deleted from the real bulletin p1 qwen candidate (39 real rows, threshold
`ceil(39 * 36/39) = 36`):

| variant | candidate_total | vs threshold (36) | outcome |
|---|---|---|---|
| complete | 39 | 39 ≥ 36 | WINNER, bound=39/39 |
| minus last 10 rows | 29 | 29 < 36 | NO WINNER (correctly rejected) |
| minus first 5 rows | 34 | 34 < 36 | NO WINNER (correctly rejected) |

Both trailing and leading truncation are caught the same way — every unmatched table-shaped
native band counts toward `native_table_rows` wherever it sits on the page, not just past a
bound range's edge, so no separate edge-row walk is needed.

### Detector defect filed separately (not this ticket's fix)

The bbox/data-row mismatch behind rounds 1-2's failed geometric designs is a `detected_table_bboxes`
defect, filed as **#639** and referenced here rather than fixed in this ticket:

| fixture | bbox y-range(s) | representative data-row y-range (rows with ≥3 numeric tokens) |
|---|---|---|
| bulletin p1 | (166.0, 258.0) | (254.2, 816.0) |
| bulletin p2 | (166.0, 242.6) | (185.3, 816.0) |
| bulletin p3 | (166.0, 265.6) | (261.9, 816.0) |
| report p1 | (171.5, 241.1) | (238.3, 356.5) |
| report p2 | (184.0, 260.6), (451.1, 513.8) | (257.7, 720.0) |
| report p3 | (183.8, 256.5), (457.1, 523.1) | (253.9, 682.0) |

On every fixture the detector's bbox ends at or before where the real numeric body starts (or
covers only its first rows before a multi-block continuation) — consistent with capturing a
caption/units box rather than the table body.

## Tests

- `PYTHONPATH=<worktree>/src ~/venvs/socr/bin/pytest tests/test_s1_structure_class_winner_corroboration.py tests/test_gh317_structure_class_floor.py tests/test_s1_structure_class_winner_gh_reachability.py tests/test_package_layering.py tests/tables/test_row_corroboration.py -q` — round 2: 139 passed; round 3: 142 passed (8 tests in the corroboration file, up from 5).
- Full suite (`pytest tests/ -q`) — round 3: 4270 passed, 4 xfailed, 0 failed (194.70s).
- `uvx ruff@0.16.0 format --check .` — round 3: 591 files already formatted (clean).
