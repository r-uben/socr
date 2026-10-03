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

## Round 4 -- region scoping DROPPED

Region scoping of the native count (rounds 1-3) was tried and removed. Astra found four bypasses,
each letting a truncated table ship because scoping RELAXES a refusal:
1. Panel truncation: extension stopped at a numeric-free panel heading (native 10 rows, heading,
   10 more; candidate emits the first 10).
2. Invented blank-stub SE rows counted without binding, masking missing rows.
3. Matching reset per block, so a repeated block re-credited the same SE bands
   (`_markdown(5) + _markdown(5)` counted 20 against 20); then three copies (10 bound + 10 unbound
   labelled = 20) did the same.
4. Same-lane bridging, and adjacent table-shaped bands that skipped the gap check, absorbed a
   second table (`_page() + _page(y0=900)`).
Each fix (lane-defined extent, bridging, pitch/caption stops, binding-coverage gate) opened the
next hole. Any rule that narrows the native count is a rule for shipping an incomplete table, so
the native side is back to main's page-wide `table_shaped_native_row_count`.

Kept: the symmetric blank-stub fix only. A blank-stub candidate row counts iff it binds to a native
band (`match_rows_monotonic`); a band credits one row on the page (blanked after use, across
blocks); a labelled row counts as on main but an exact repeat (label + tokens) is not counted again.
`row_shape_min` stays the min over labelled counted rows. No extent, bridge, pitch, caption or
coverage logic remains. Never more lenient than main except for corroborated SE lines.

Census (40 refused candidates): 14 clear, 26 still refuse, 0 refusals new vs main. (Region scoping
had cleared 24; the other 10 were the price of the bypasses.) Cleared pages viewed: ghost p13,
huynh p32, bybee p31 (render viewed in the original census notes) -- all complete tables.
Cleared-and-incomplete: 0.

Mutations (external copy, canary): blank-stub never counts 1 killed; unbound blank-stub counts 3;
bands reusable across blocks 2; no labelled dedupe 1. Baseline 59 passed.
`tests/tables/test_gh703_text_table_dominance.py` is back to origin/main (no edits needed).

## Round 5 -- the symmetry applied to the SE credit

Astra: 10 coefficient rows of three numbers, each followed by a one-number SE row; candidate emits
five pairs. `row_shape_min` is 3, so native counts the 10 coefficient bands only, while the
candidate counted 5 + 5 bound SE rows = 10 and passed (main refuses). Fix: a bound blank-stub row is
credited only if its native band passes the test `table_shaped_native_row_count` applies (>=
`row_shape_min` numbers, not a column-index legend). `row_shape_min` is computed first (labelled rows,
else the bound blank rows). Pin: Astra's exact page. Mutants: shape condition removed 1 killed;
width test dropped 1 killed; baseline 60 passed.
Census: 13 clear / 27 refuse / 0 new vs main (huynh p32 no longer clears: its SE rows have fewer
numbers than its coefficient rows). Cleared pages are a subset of those viewed before
(ghost p13, bybee p31 etc.): all complete; cleared-and-incomplete 0.

## Round 6 -- repeat detection by native band, not label

Astra: the first five pairs three times under different labels (Var0 / Variable0 / VAR0) counted
15 labelled + 5 bound SE = 20 and passed; main counts 15 and refuses. The exact-label dedupe is
removed. A labelled row that does not bind but reproduces (contiguous token run) a native band that
another candidate row already consumed is a repeat and is not counted; a labelled row that binds to
nothing at all still counts, as on main (OCR drift). Pin: Astra's case as separate blocks and as one
block. Mutants: repeat rule off 2 killed (3-copy + Astra); bands reusable 4; shape condition 1.
Census: 13 clear / 27 refuse / 0 new vs main (same 13 cleared pages as round 5).

## Round 7 -- keep only the provably safe subset

Astra: unbound labelled repeats whose band lies before the block's first match stay unconsumed and
inflate the count (14 repeats of row 0, plus bound SE rows -> 20; main 15). Repeat detection (rounds
5/6: label dedupe, then consumed-band matching) is removed. Rule now: main's counting EXACTLY, unless
every labelled candidate row binds to its own native band (unique across the page) and each band
passes the native countability test (>= row_shape_min numbers, not a legend); only then are
blank-stub rows credited, each under the same bound-and-countable rule. Any unbound or shared-band
labelled row falls back to main's count, so the branch is never more permissive than main on that
page. Pin: Astra's r6 case plus the r5 / earlier reproducers.
Census (40): 12 clear (of the previous 13; bybee p37 drops), 28 refuse, 0 new vs main.
Mutants: drop "every labelled row bound" 3 killed; blank band need not be countable 1; blank
unbound credited 3; bands reusable 4; never credit 1. Baseline 62 passed.

## Round 8 -- full-sequence binding

Astra: `match_rows_monotonic` binds a contiguous SUBSET of a band, so a candidate with whole columns
dropped (only column (1); 20 rows, 40 of 60 values missing) bound all 20 rows and was accepted while
main refuses. On the superset path only, a row now binds to a band only if its numeric tokens equal
the band's FULL token sequence (same count, same order, same normaliser); anything partial sends the
page to main's count. Main's path is untouched. Pin: Astra's r7 case.
Census (40): 11 clear (of the previous 12; Giroud p10 drops), 29 refuse, 0 new vs main.
Mutants: subset binding allowed 1 killed; every-row-bound dropped 3; never credit 1. Baseline 63.
