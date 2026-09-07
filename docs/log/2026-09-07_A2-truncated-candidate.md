# TICKET-A2 (#645): a truncated candidate never beats a complete one

Branch `fix/645-truncated-candidate` off `main@393ec6d` (post-A1c/#644), worktree
`~/repos/.worktrees/socr-a2`.

## Problem

Bulletin p3 (`5.5 Counterparts to M3`): qwen's output ended mid-number
(`| 2019 | 364.2 | 7,05`). 34/389 numbers plus 55 wrong values shipped as a
WARNING, over gemini's complete 389/389 reading, because the S1 strict
grid-authored pool held ONLY the truncated qwen candidate — gemini's complete
reading never cleared strict-grid shape — so it shipped via
`structure_class_model_table_kept` and A1b's row-corroboration fallback never
ran (the strict pool wasn't empty).

Reproduced live on `main+A1c` (2026-09-07,
`/tmp/a1c/ecb-meetings-2021-economic_bulletin-p127-129`, page 2): two cached
qwen candidates, complete 414/417 (`status success`) and truncated 14/417
(`status warning`, ends in the flag note). Same shape.

## Fix

`table_truncated` in `structure_check.py`, evaluated on raw markdown (+
native words when available) before strict parsing discards malformed rows.
Two terms, either sufficient:

- **(a) style break**: the candidate's own body rows are mixed —
  fully-bordered rows plus one final row that breaks that established style
  (no closing pipe). A candidate whose rows are ALL consistently unterminated
  is its own formatting convention, not truncation, and is never flagged.
  Abstains with fewer than 2 body rows (nothing to compare style against).
- **(b) row shortfall**: candidate's numeric body-row count falls short of
  the native table-shaped row count (`row_corroboration.
  table_shaped_native_row_count`, the same function A1b's
  `_row_shape_reconciliation_ok` uses) by more than A1b's own
  `ROW_CORROBORATION_MIN` (36/39) allowance permits.

A candidate flagged `table_truncated` is excluded from both S1 pools
(`_strict_grid_authored_pool` and the row-corroboration fallback) via
`_truncated_grid_reading_ids` — but only when at least one OTHER candidate
for the page is not truncated. If every candidate truncates, or it's the
only candidate, nothing is dropped (there's no complete reading to prefer)
and it still ships, flagged. Each drop emits a `candidate_truncated {engine}`
audit event (`audit_log.py` rank 6, alongside the other S1-story kinds) and a
`tables_trust.py` distrust-kind entry.

Precedence in `table_output_defect` (unchanged shape, one insertion):
1. `table_emission_defect`
2. `table_content_defect`
3. **`table_truncated`** (new)
4. `structural_gate_fires`
5. `header_cut.header_cut_verdict`

## Header-row inflation bug found and fixed along the way

Term (b) reuses `table_shaped_native_row_count` over the whole page's native
words. On synthetic fixtures with a numeric-looking header row (e.g. year
columns `1997 2002 2007`) that isn't caught by `is_column_index_row` (which
only excludes sequential-digit legends like `1,2,3`), that header band counts
as a 4th "table-shaped" native row — while the candidate's own header row IS
correctly stripped by `numeric_body_rows`. Result: `native_table_rows=4` vs
`candidate_rows=3`, `ceil(4 * 36/39) = 4 > 3` → false-positive truncation,
which (by precedence) pre-empted the header/shape defects
`tests/test_header_cut.py`'s four `TestWiredIntoTheShippingGate` tests and
`tests/test_native_table_verifier.py::TestTR4RowCount::
test_page_numerals_excluded_by_yband_exact_pass` were actually testing for.

Confirmed via A1b's own docstring in `manifest._row_shape_reconciliation_ok`:
measured at ratio 1.000 (exact match, no strays) on all six real ECB
fixtures — this asymmetry does not manifest on real production data, only on
small synthetic tables where a header row happens to be numeric and
column-aligned. This is the SAME shared mechanism GH-643 ("row-shape
reconciliation wrongly keeps...") already tracks; fixing the shared function
itself is out of scope for A2.

**Fix applied at the term-(b) call site only** (not the shared function, to
avoid destabilizing A1b's already-measured 1.000 ratios): added
`_STRAY_HEADER_BAND_ALLOWANCE = 1` — term (b) now also requires the absolute
row gap to exceed one stray band (`count < native_table_rows - 1`) before the
ratio check even applies. Verified by hand-trace against:

- header_cut fixture: native=4, candidate=3, gap=1 → does not fire (fixes
  the regression).
- `test_row_shortfall_past_allowance_is_truncated` (20 native rows, drops
  2 → 18): gap=2 → still fires.
- `test_row_shortfall_within_allowance_is_not_truncated` (drops 1 → 19):
  gap=1 → does not fire (unchanged, already passed on ratio alone).
- Real bulletin p2/p3 fixtures: gaps well past 1 → still fire.

## A second regression: cascade loop-free invariant

`tests/test_r7_winner_kind_tags.py::test_cascade_is_loop_free_so_exactly_one_
ending_runs` AST-walks `manifest._select_page_output_tagged` and asserts no
`for`/`while` node exists in its body, since a loop there could break the
"exactly one ending per page" guarantee `SelectionProvenance` depends on. My
original call site emitted `candidate_truncated` events with a `for
truncated_engine in truncated_engines:` loop written directly inside that
function. Fixed by extracting the loop into a standalone helper,
`_truncated_candidate_events(page_num, truncated_engines)`, which builds the
list via a comprehension (a comprehension is not an `ast.For`/`ast.While`
node) and lives outside `_select_page_output_tagged` entirely. The call site
is now a single `state.events.extend(_truncated_candidate_events(...))`.

## Test results

- `tests/tables/test_structure_check_truncated.py` (new, 24 tests) +
  `tests/test_s1_structure_class_winner_corroboration.py` +
  `tests/test_header_cut.py` + `tests/test_native_table_verifier.py` +
  `tests/test_r7_winner_kind_tags.py`: **101 passed**.
- Full suite: `PYTHONPATH=src ~/venvs/socr/bin/pytest tests/ -q` →
  **4298 passed, 4 xfailed**, zero failures, zero regressions.
- `uvx ruff@0.16.0 format --check .`: 3 files needed reformatting
  (`manifest.py`, `row_corroboration.py`, the new test file) — reformatted,
  re-checked clean (594 files formatted), re-ran the affected 101 tests to
  confirm no behavior change from formatting alone.

## Files changed

`src/socr/tables/structure_check.py`, `src/socr/core/manifest.py`,
`src/socr/core/audit_log.py`, `src/socr/core/tables_trust.py`,
`src/socr/tables/row_corroboration.py`, `src/socr/tables/reconcile.py`,
`tests/tables/test_structure_check_truncated.py` (new).

## Live pipeline verification

Ran `PYTHONPATH=src ~/venvs/socr/bin/socr process
~/Data/socr/census-ecb-2026-09-06/in/ecb-meetings-2021-economic_bulletin-p127-129.pdf
-o /tmp/a2/` on the branch, Ollama available. The run was killed partway
through page 3 by the host's OOM pressure (`bdpisl036` background task,
status `killed`, "system is running low on memory") — the `socr` process
itself was gone, not a socr-internal failure. Per team-lead's instruction,
**not re-run**; page 3 is reported as not scored, and the decision to re-run
once memory is available is team-lead's.

Pages 1-2 survived (per-page ledger flush) and were scored per-page: source
numbers from `pdftotext -layout -f N -l N` on the original PDF, matched
against that page's shipped `NNN.md`, normalized (commas/percent/trailing-dot
stripped) exactly as `census_score.py` normalizes, one-shot ad hoc script
since the shared scorer takes a whole-document pdf/md pair and this run only
has two of three pages.

| page | status | failure_mode | engine | source numbers | matched | missing | recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | warning | header_binding_unverified | qwen | 497 | 496 | 1 (`20`) | 0.998 |
| 2 | warning | header_binding_unverified | qwen | 417 | 416 | 1 (`35.5`) | 0.998 |
| 3 | — run killed by memory pressure, not re-run — | | | | | | |

Both surviving pages are effectively complete (single-token misses, not the
34/389-shipped defect this ticket targets). Note this run's page 2 does not
reproduce the specific truncated-vs-complete qwen pair documented in the
Problem section above (that pair was observed in an earlier cached run,
`/tmp/a1c/...-p127-129`, 2026-09-07) — Ollama model output is stochastic, and
this run's page 2 shipped a single non-truncated qwen candidate rather than
the truncated/complete pair. No document-level `audit_log.json` exists for
this run (it is written at assemble time, which never ran given the crash),
so no `candidate_truncated` event count is available for this specific
attempt; the deterministic unit/integration tests above (built from the
actual cached truncated-candidate JSON for this same fixture) are the
guard's real evidence, not this one stochastic partial run. Page 3 (the
ticket's own headline defect, `5.5 Counterparts to M3`) is unverified live
until team-lead reruns with memory available.
