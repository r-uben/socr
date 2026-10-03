# 2026-10-03 -- #988 table_truncated row-shortfall counts like with like

Branch `fix/truncated-shortfall-symmetric` from origin/main 3498cb27. Issue r-uben/socr#988.

## Change

- `row_corroboration.numeric_body_rows(rows, *, include_blank_stub=False)`: opt-in to keep
  blank-first-cell rows (SE / t-stat lines). Default unchanged (A1a / manifest callers untouched).
- `structure_check._truncated_row_shortfall`: candidate side counts blank-stub rows;
  native side counted by `_native_table_rows_in_candidate_region`.
- `_native_table_rows_in_candidate_region`: binds the candidate's rows to baseline bands
  (`match_rows_monotonic`), spans first..last bound band, extends outward only through ADJACENT
  table-shaped bands (so rows missing at either end stay inside the region), then counts with
  `words_in_region` + `table_shaped_native_row_count`. No bound band -> page-wide count (never
  relaxes on evidence it could not locate). No new constants.
- Final-row term, lane gate, `_STRAY_HEADER_BAND_ALLOWANCE`, `ROW_CORROBORATION_MIN`: unchanged.
- NOT changed: `manifest._row_shape_reconciliation` (same page-wide asymmetry; separate gate,
  its docstring records why it avoids regions). Follow-up if wanted.

## Test change to an existing pin

`test_real_boe_p1_difference_pin` demonstrated the #703 lane gate by ungating term (b). Region
scoping alone now stops that page truncating, so the ungated arm also needed the page-wide count
restored to keep the pin about the lane gate. No assertion weakened.

## Re-measurement (census: 93 override pages, 40 refused candidates on 39 pages)

Script: scratchpad remeasure.py (old logic re-implemented inline, new = production code).
- Still refused before: 40. After: 16. Cleared: 24. New refusals: 0.
- Cleared with page numeric recall < 0.97: lanza p22 (0.55), bybee ghost p13 (0.50), bernanke p17
  (0.90). All three viewed at render: tables complete (lanza: Rows/Columns table; ghost p13:
  CFO/AAII correlations + t-stats; bernanke p17: Tables V and VI complete). Recall is low because
  the page's other numbers are chart ticks / prose / watermark. Cleared-and-incomplete: 0.
- The 16 still refused (14 recall >= 0.98): candidate cells are `$0.96^{***}$` / starred values,
  which `_is_genuine_numeric` rejects, so those rows are dropped from the candidate count
  (Arslanalp-Eichengreen x6, xiao x2, others). A third asymmetry, in the shared numeric
  predicate; out of scope here. bernanke p6 (recall 0.84, ladder withheld T-II, invented cells)
  and Barrot p25 (0.91) are the only two with real recall gaps.

## Mutations (external copy, canary socr.__file__ inside copy, 3 suites)

M1 no blank-stub: 2 killed. M2 page-wide: 1. M3 no downward extension: 14. M4 no upward: 1.
M5 no-bound relaxes: 1. M6 extend through anything: 1. Baseline 57 passed.

## Round 2 (Astra rejected 1ea33233: two bypasses)

1. Panel truncation: adjacency-only extension stopped at a numeric-free panel heading and the
   candidate's own rows alone set the extent. Now the extent grows through adjacent table-shaped
   bands AND bridges a run of non-table-shaped bands when the next table-shaped band beyond has
   ALL its numbers in the lanes (x0/x1, `reconstruct._LANE_X_TOL_PT`) of the bands the candidate
   bound. Native geometry decides the far end; tick/running-head numbers elsewhere are not bridged.
2. Invented SE rows: a blank-stub candidate row counts only if it binds to a native band
   (`match_rows_monotonic`). Labelled rows count as before.
3. Scoping applies only when bound bands >= ceil(counted rows * ROW_CORROBORATION_MIN); otherwise
   the page-wide count stands (one stray binding cannot pick the region).
4. `row_shape_min` is again the min over LABELLED counted rows (a numeric header with a blank stub
   had lowered it and produced a new refusal on Eichengreen p21 that main accepts).
Known limitation kept: citation rows on a text page still expose term (b) (bridge reaches them);
the #703 test pinning that is restored to its original assertion.

Census re-run (40 refused candidates): 24 cleared, 16 refuse, 0 refusals new vs main.
Cleared set changed by one: acosta p35 (complete table, starred coefficient rows are not
counted -> refused as before this fix) out, huynh p32 (viewed: complete) in. bernanke p17, ghost
p13, lanza p22 stay cleared. Cleared-and-incomplete: 0.

Mutations (8, external copy, canary): blank-stub-never-counts 2 killed; unbound-blank-stub-counts 1;
always page-wide 1; scope-on-any-binding 1 (needed a fixture fix: ticks adjacent to the table
were merged by adjacency); no bridge 1; bridge without lane check 1; no adjacent extension 7.

## Round 3 (Astra rejected fe26b80)

- Native bands are consumed uniquely across blocks (a band bound by one block is blanked for the
  next), so a repeated block cannot re-credit the same SE bands. Pin: `_markdown(5) + _markdown(5)`.
- A bridge stops at a `Table` / `Figure` lead word and when its vertical span exceeds one table row
  pitch per bridged band plus one (pitch = largest spacing between table-shaped bands inside the
  bound span; no pitch -> no limit). Pins: two tables in the same lanes (wide gap; caption).
- Census: still 24 cleared / 16 refused / 0 new. Mutations: bands reusable 1 killed, no gap rule 1,
  no caption rule 1 (after making the caption fixture's gap pass the pitch rule), bridge limits
  removed 1, no bridge 1.
