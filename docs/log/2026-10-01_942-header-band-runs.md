# GH-942: header_band_missing fires on a header row that splits into runs

Branch `fix/942-header-band-runs`, cut from origin/main 0567718 (ancestry verified).

## Change

`src/socr/tables/ship_gate.py`, inside `header_band_missing_faults` plus a new helper `_run_count`
(and the `_median_word_gap` import from `reconstruct`):

- Main's clause (region words on distinct lanes) is kept.
- OR-ed with a run clause: the candidate row's in-extent words split into `>= _MIN_LANES_PER_ROW`
  runs at gaps wider than `ALIGNED_RUN_GAP_MAX_WORD_SPACES` x `_median_word_gap`. The gap unit is
  taken from all page words (flattened `src_rows`; bare 5-tuple words are skipped, so the
  `direction_unavailable` contract is unchanged). Candidate row, reach, numeric-free test and the
  per-block absence test are shared and unchanged. No new constant, no new gate input.
- The OR is required (design.md): replacing main's clause flips 2 DEFER pages to SHIP.

## Census (127 pages, frozen inputs `gh942/inputs.pkl`)

Baseline is origin/main's `ship_gate.py` in a copy of this tree, so only this change differs.

- `header_band_missing` fires: 18 -> 46. Removals: 0 (asserted `main_set <= branch_set`).
- Pages that DEFER on any predicate: +14, removals 0, other predicates unchanged on all 127.
- The 14 new DEFERs match design.md's list: 13 real column-label losses (Boukus 38, Bybee 27,
  Cieslak 51 and 52, De Fiore 33, Eskildsen 15 and 70, Fan 52, Perico-Ortiz 36 and 41,
  Piller 16, Sarkar 50, Siano 45) and 1 beneficial (Gong 46, garbled header). 0 false.
  The 14 are the same pages the design labelled against the renders; the 14 other added fires
  land on pages that already DEFER.
- Residual misses (design): Boukus 39 and Gurkaynak 46.

## Tests

`tests/test_gh942_header_band_runs.py` (12 tests): multi-word and right-aligned headers that main's
lane clause misses now fire (difference pin, with the same page carrying the header quiet); 2 runs
quiet vs 3 fire; run-gap bound; controls (header in grid, one-run caption left out, numeric row);
main's clause still required (page without prose, one run, still fires); absence per block
(two panels, header repeated in panel B).

## Mutations (external copy of src + tests + pyproject, canary on `socr.__file__`, anchor count 1)

Baseline passes (63). Each of these fails: floor 3 -> 2; run clause removed; OR removed (run
clause only); K -> 1000; K -> 0; absence page-level.
