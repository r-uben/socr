# 2026-10-02 GH-949: DEFER a grid with no geometry that carries fewer rows than its source

Base: origin/main 5e2c358.

## Problem

`_table_geometry` needs two unique pairs. A grid holding only a table's shaded highlight row
has one, so geometry is None and every geometric predicate stays silent. On liu_cao_flake
p49, p50, p62 the grid shipped SUCCESS while the rest of the table came out as loose lines.

## Measurement (127-page frozen census, `inputs_now.pkl`, which includes liu p49/p50/p62 and beckmann p79)

- Blocks with no geometry on the census: 3. All three are the liu pages.
- Predicate fires: 3 pages, all liu_cao_flake (p49: source 5 rows vs grid 2; p50: 9 vs 2; p62: 9 vs 2).
- Classified by viewing renders: all 3 are real truncated tables (shaded row in the grid, the
  Turns/Order/Duration/FE/Observations rows outside it). False alarms: 0. Fires outside liu: 0.
- Other pages that fire: none. Other predicates fire on these 3 pages: none (that was the bug).
- beckmann p79 is NOT this bug: its block HAS geometry (5 unique pairs), so it is a different
  defect and stays out of scope.
- Monotone: the predicate only appends faults; 0 removals versus main by construction, and the
  census shows the 124 other pages with an identical fault list.

## Change

`src/socr/tables/ship_gate.py`: `geometryless_block_faults`, predicate `geometryless_block`,
wired last-but-one in `native_ship_gate`. For a block whose geometry is None and that has a
paired row with >= 2 numeric words: lanes are that row's numeric x-clusters; the source region is
the anchor plus contiguous rows reaching `_MIN_CORE_LANES` of those lanes, each within
`_PANEL_GAP_ROWS` page-median row pitches of the last. Fires when the region has more rows than
the grid has rows with >= `_MIN_CORE_LANES` numbers. Reuses existing constants; no new thresholds.
Words are only indexed at 0, 1, 4, so 5-tuples work (tested, width 5 and 8).

## Tests

`tests/test_gh949_geometryless_block.py` (9): lone highlight-row grid fires; identical grid over
a one-row source is quiet; label-only padding does not stand in for numeric rows; complete grid
quiet; geometry-present block untouched; only-adds-faults difference; reach bound; 5-tuples.
The gate is pure, so nothing depends on a provider or ollama.

## Mutations (external copy of src + tests + pyproject, `socr.__file__` asserted inside the copy,
uncapped `count == 1` anchor asserted before every edit, baseline control green: 9 passed)

| mutant | result |
| --- | --- |
| unwire from `native_ship_gate` | 7 failed |
| `if len(region) > carried` -> `if False` | 7 failed |
| remove `geo is not None: continue` | 1 failed |
| remove the reach `break` | 1 failed |
| `carried = len(block)` | 1 failed (needed the label-padding test; first run survived) |

## Follow-up

Beckmann p79 (7 of 39 table lines in the grid, geometry present) is a separate defect.

## Review round (Astra, ACCEPT-WITH-FIXES)

- Region bound: the walk used to extend one reach per hop and count any lane-wide row, so a
  neighbouring same-column table or a prose line holding years could count. It now walks every
  source line outward and stops at the first that is not a multi-column numeric row (a label may
  precede the first number; after it only cell numbers or no-value dashes) or lies beyond one
  reach. Cell numbers tolerate `(...)`, significance stars, a leading sign (`_is_cell_number`);
  the first cut without that lost p49 and p50 because the starred and parenthesised rows ended the walk.
- Census re-run: the 3 liu pages still fire, 0 other fires, 0 removals (append-only).
- Tests: neighbour-table-behind-a-caption control and prose-with-years control (both quiet, each
  paired with the firing twin); starred/parenthesised cells extend the region; the monotonicity
  test now has a pre-existing fault (`direction_unavailable`) and asserts it is kept; the label
  padding test is renamed to say it pads the OUTPUT.
- Mutations of the new rule (external copy, baseline 12 green): drop the stop rule 4 failed, drop
  the reach term 1, accept prose after the first number 1, plain numbers only (no star/paren
  tolerance) 1; earlier mutants still killed.

## Known limits

- A lone numeric row plus many omitted label-only SOURCE rows stays quiet (region = carried = 1).
- A neighbouring table with no line between it and the block (gap under one reach, no caption)
  still counts.

## Second review round: split pins

The starred/parenthesised test is now two independent pins (starred-only row, parenthesised-only row). Mutants in an external copy (baseline 13 green): `_is_cell_number` with the stars removed from its strip set fails 1; `_is_cell_number` rejecting any text containing `(` fails 1. Finding: `()[]` in `_CELL_WRAP` is redundant, because `_is_source_number` already accepts `(-2.19)` and `(1.40)`; a mutant that only removes `()[]` from the strip set survives and is equivalent. The parenthesis pin guards the behaviour (parenthesised rows extend the region), not that constant.
