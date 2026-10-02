# GH-958: scattered panel heading and header-over-empty-column (ship gate, DEFER-only)

Branch `fix/958-scattered-heading-header-lane`, from origin/main 760b5b33.

## What changed

Files: `src/socr/tables/ship_gate.py`, `tests/test_gh958_scattered_heading_header_lane.py` (new),
`tests/test_gh917_text_in_numeric_column.py` (one test renamed and re-documented).

**(A) `text_in_numeric_column`.** The panel-label exemption (row's first non-empty cell left of the
first numeric column, row before the last data row) no longer covers a row when all hold:
at least `_MIN_CORE_LANES` text cells in numeric columns; the row's joined cells (NFKC, whitespace
stripped) equal ONE whole source line; that line is one run (`_is_one_run` in `_page_word_space`
units, zones as in `prose_in_header`). No measurable word space: abstain, exemption stands. The
one-run clause is required: without it fama p469 row 19 (`5-Yr SR | High ... Low`, a correct
positioned sub-header) fires falsely. `text_in_numeric_column_faults` takes optional
`words` / `src_rows` / `geos` (keyword); `native_ship_gate` passes them.

**(B) `header_over_empty_column`** (new predicate, own function; smaller than extending A). A header
cell (row above the first data row) over a column blank in every data row, with an adjacent numeric
column blank in every header row. Output-side only, GH-917 classifier and data-row rule. No
lane-centre clause: it misfires on stock_watson p43 (two-line header spanning a value and its
(n) column, cosmetic).

`test_panel_label_spanning_into_numeric_columns_is_exempt` passes only because its fixture has no
source line; its origin (fama p782) is the defect shape. Kept, renamed
`..._with_no_source_line_is_exempt`, re-documented. The difference pin is in the new test file.

## Census (127 frozen pages, 760b5b33 vs this tree)

Inputs (frozen copy): `/Users/rubenffuertes/.local/state/socr-housekeeping/gh958/impl/inputs_now.pkl`
(from `/Users/rubenffuertes/.local/state/socr-housekeeping/triage4/inputs_now.pkl`).
Scripts and outputs: `/Users/rubenffuertes/.local/state/socr-housekeeping/gh958/impl/{census.py,diff.py,before.json,after.json,mutate.py,suite.log}`.
Prior measurement: `/Users/rubenffuertes/.local/state/socr-housekeeping/gh958/{NOTES.md,measure.py,pin.py,m2.json}`;
audit: `/Users/rubenffuertes/.local/state/socr-housekeeping/ship-audit/AUDIT.md`.

- A: 3 fires (fama p475, fama p753, woodford p787 already DEFER); SHIP->DEFER: fama p475 (1).
- B: 6 pages carry `header_over_empty_column`; SHIP->DEFER: harren_kilic_zhang p66 (1).
- SHIP 12 -> 10, 0 removals (every predicate set only gains faults).

## Mutations (external copy of src + tests + pyproject, `socr.__file__` canary, uncapped anchor count 1, baseline 49 passed)

A never judges true: 2 killed. A without run clause: 2 killed. A min-lanes 2 -> 1: 1 killed.
B never fires: 3 killed. B without adjacent-headerless: 5 killed. B without empty-column: 3 killed.

## Tests

New file 17 tests (initially logged as 11); 917 file 37 tests (initially logged as 38). Full suite, default OLLAMA_HOST, nohup, one run:
6207 passed, 2 skipped, 4 xfailed. Hermetic: no provider call (gate and `plan_native_table` only).

## Review follow-up (Astra ACCEPT-WITH-FIXES, coordinator ruling; rules not narrowed)

- A: `Panel A: | Small | firms` between data rows with a single-run source line is the defect shape
  (fama p475), pinned as an intended fire; the merged single-cell label stays quiet.
- D: the EXACT original row `Big, low-profitability | growth firms: | dSM | < 0` was measured with a
  single-run source line backing it: it FIRES (`text_in_numeric_column`). Right for fama p782: that
  row is the sentence scattered over numeric cells, the shape the audit called wrong.
  Markdown-only it stays exempt (the renamed GH-917 test).
- B accepted false DEFERs (cost: one model read each): a deliberately unfilled column with a header
  (`Year | Forecast | [blank] | Actual`, headerless neighbour), and a group heading over a spacer
  column. Both are indistinguishable from the one-lane-off defect by the grid alone. Pinned for the
  first; the second is the same grid shape.

## Cubic review (PR #959)

- P2a: the scattered-heading match now only accepts source lines inside the block's own table zone
  (first/last core row +- `_header_reach`); a duplicate line elsewhere on the page no longer backs a row.
- P2b: the check abstains for the whole page when any block has no geometry (its rows cannot be excluded
  from the `_page_word_space` calibration). Chosen over excluding those rows: no zone exists to exclude.
- Census re-run on `impl/inputs_now.pkl`: unchanged. SHIP 12 -> 10, fama p475 and harren p66 DEFER, 0 removals.
- Mutations (external copy, canary, anchor count 1, baseline 55 passed): dropping the zone scope is killed by
  `test_a_duplicate_line_outside_the_table_does_not_back_the_row`; dropping the geometry abstain (skipping
  geometry-less blocks in calibration) is killed by `test_a_block_without_geometry_makes_the_check_abstain`.
  The earlier A/B mutants are still killed.
