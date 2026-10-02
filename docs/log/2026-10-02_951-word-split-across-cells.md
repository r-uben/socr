# 2026-10-02 GH-951: DEFER a grid row carrying a source word split across two cells

Base: origin/main 8597aa6 (ancestry verified).

## Problem

Native-first shipped grids whose caption or notes paragraph is cut mid-word across cells
(levy p105, faust p46) and no predicate fired.

## Measurement

Census data: `/Users/rubenffuertes/.local/state/socr-housekeeping/gh951/inputs_now.pkl`
(127 pages; probe scripts `census.py`, `cmp.py`, `joined.py` and `ev_main.json`, `ev_var.json`
in the same directory). Re-run from THIS implementation (`native_ship_gate` on every page,
with and without the new predicate in the same process), not from the probe:

- pages 127; fires 13 pages, all real splits (faust 44, faust 46, bybee 36, cieslak 63,
  gow 48, cook 21, hansen 29, hansen 30, gong 46, levy 105, barry 19, jiang 50, sr99 12).
- new DEFERs (page shipped before, ships no more): 3 = faust 46, levy 105, sr99 p12.
  The other 10 pages already carried a fault from another predicate or did not ship.
- removals: 0 (fault list with the predicate stubbed out is a subset of the full list on every page).

## Change

`src/socr/tables/ship_gate.py`: `word_split_across_cells_faults`, predicate
`word_split_across_cells`, wired as one line at the end of the fault list before
`foreign_direction_faults`. Fires when, in one row, the last token of a non-empty cell
plus the first token of the next non-empty cell, NFKC-normalised and joined with no space,
equals a source word that is not a token anywhere in the block and for which
`_is_source_number` is false. No new constant. It reads only `w[4]`, so short tuples work.

Decisions:
- A hyphenated compound cut at its hyphen (`well-` + `known`) FIRES: the source keeps it as
  one word and a cell does not otherwise end in a hyphen.
- Known hole: a word that also occurs whole elsewhere in the block stays quiet.
- "Grid" means the block, as in the probe that produced the numbers.

## Tests

`tests/test_gh951_word_split_across_cells.py` (9): split vs whole-word difference (unit and
end to end), hyphen compound, numeric split quiet, word-elsewhere known hole, ligature folding,
no join across rows, short tuples/empty input, only-adds-faults with `line_dirs=None`.
Pure gate; no provider or ollama dependency.

## Mutations (external copy of src + tests + pyproject, canary asserting `socr.__file__` is
inside the copy, uncapped `count == 1` asserted before each edit, baseline 10 passed)

| mutant | result |
| --- | --- |
| unwire | 2 failed |
| drop `not in present` | 1 failed |
| drop number guard | 1 failed |
| drop `in source` | 7 failed |
| drop NFKC | 1 failed (first run survived; ligature test added) |
| join last cell to first cell of the same row | 1 failed |

## Review round (Astra, ACCEPT-WITH-FIXES): the source-word lookup was page-wide

Blocking point: `Pre | tax`, `In | come`, `non | linear` fired whenever `Pretax`, `Income`,
`nonlinear` occurred anywhere else on the page. Fixed by making the evidence positional:

- the row's tokens must cover ONE whole source line in order (`_aligned_cuts`): each source
  word is one token, or two tokens of different cells joined. A cut fires only at a join the
  line itself places. So `non | | linear` fires only if one source word `nonlinear` is where
  the cells meet, and `well- | known` only if that word is `well-known`.
- either fragment being a number disqualifies the pair (`5 | kg` over source `5kg` is quiet).
  The old check on the joined word is gone: it was redundant (its mutant survived).
- a row wrapped over several source lines now abstains (documented hole).

Census re-run (same data, `/Users/rubenffuertes/.local/state/socr-housekeeping/gh951/inputs_now.pkl`):
12 fire pages, 0 removals; faust 46, levy 105 and sr99 p12 still DEFER (3 new DEFERs). The
12 are a subset of the 13 judged real earlier: 12 real, 0 false. Lost: cook_kazinnik p21,
whose notes paragraph wraps over several source lines (a true split, now abstained).

Tests: 17 in the file, adding fixed-source controls (collision, number+unit both sides,
unrelated source perturbation, cross-row positive counterpart, hyphen compound,
whole-line coverage, one-cut-per-line). Mutations in an external copy (src+tests+pyproject,
canary, uncapped count 1, baseline 18 passed): unwire, drop not-present, drop left/right
fragment number guard, drop positional rule (5 failed), any-cut, partial line coverage,
drop NFKC: all 8 caught.
