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
