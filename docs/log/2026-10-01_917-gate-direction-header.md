# GH-917 PR A: foreign direction and dropped header band in the native-first ship gate (2026-10-01)

Branch `fix/917-gate-direction-header` from origin/main c0c67e4 (ancestry verified). Design:
`~/.local/state/socr-housekeeping/gh917/design-917.md` (Fable); Astra consult settled O1 now, with the
rowizer fixes as their own tickets (#924 split date, #925 header band). The rotated quarantine (#918) stays;
nothing here lifts it. The rowizer is untouched.

## What changed

`src/socr/tables/ship_gate.py` (both predicates DEFER only):

1. **`foreign_direction` (P7).** Per OUTPUT TABLE BLOCK, a source word is carried when its text occurs in a
   cell of that block (`_CellText.has_word`). Any two carried words whose line directions differ by at least
   `_direction_tolerance(extent)` is a fault, so a tie between two directions defers and nothing chains.
   - **One frame.** Directions are PyMuPDF line `dir` vectors of the same page frame; the comparison is the
     angle between two vectors, `atan2(|cross|, dot)`. A direction is a vector, not an axis: upside down
     (pi) is foreign. Page rotation moves both vectors alike, so the angle is frame independent.
   - **Tolerance, derived and named.** `_direction_tolerance(extent) = atan2(_snap(), max(extent, _snap()))`:
     the angle at which a line drifts by one lane snap (`_LANE_X_TOL_PT * _LANE_SNAP_MULT` = 18 pt) across the
     block's x-extent. About 2.7 degrees on a 385 pt table. Float jitter is orders of magnitude below it,
     a stamp or rotated head far above, and it is not a bucket (pairwise, no merging).
   - **`_CellText.has_word` is substring presence, not occurrence attribution.** A foreign `0.2` anywhere on
     the page is "carried" because `0.253` contains it, and a one-letter word is carried by nearly any cell.
     Conservative over-DEFER by design; pinned by `test_membership_is_substring_presence_a_documented_over_defer`.
2. **Failure contract (Astra, highest priority).** `native_ship_gate(words, markdown, line_dirs=None)`:
   - `None` = deliberately not supplied (unit tests); P7 is skipped.
   - A `LineDirections(dirs, fault)` is what production passes. Failed extraction (`fault` set), an empty map,
     a plain dict / wrong type, a carried word with no `(block, line)` key (including words with no block/line
     indices), or an unusable value (zero vector, NaN, None, wrong shape) each produce a
     `direction_unavailable` fault: DEFER, recorded in the audit event, never a silent skip.
   - `line_directions_for_page(page)` never raises; it returns the fault. At the upright emit site it sits
     inside the `open_pdf` block whose exception REFUSEs ("text layer unreadable"), and cannot inherit that
     REFUSE because it cannot raise (pinned: patched `get_text("dict")` failure gives DEFER + recorded
     `direction_unavailable`, not REFUSE, not SHIP).
   - `TEXTFLAGS_WORDS` is load-bearing. Pinned on an image-block fixture (images before and after the title):
     every word has a key and the key's own line direction. Without the flag the test fails.
3. **`header_band_missing` (H-V3).** A source row above a block's first core row, within
   `_PANEL_GAP_ROWS` (5) pitches (the outward reach `data_row_missing` already uses), not a paired row, with no
   number inside the table's x-extent, with at least one word absent from the block's cells, and whose
   lane-region words (x0 at or right of first lane minus one snap; the stub column is outside) number at least
   `_MIN_LANES_PER_ROW` (3), each x-centre within the snap of a lane, no two on one lane. Zero source rows of
   that shape means zero fires. One difference from the design script: it zipped `table_spans` output with
   `anchors.per_block`, which misaligns when a block has no geometry; this uses `_table_geometry` per block
   (same lanes and core rows). It reproduced the design's counts exactly.

Plumbing (`line_dirs` into `plan_native_table` then `native_ship_gate`), three sites, not two:
`attempt_rotated_native_table` (holds the page; the orchestrator rotated site calls it),
`_plan_native_table_first` upright branch, and `_repair_native_table_cells` (a third `plan_native_table` caller
that can SHIP after a cell repair; it re-opens the page and would otherwise have run the gate with P7 silently
off). All three build the map with `line_directions_for_page`.

## Tests (`tests/test_gh917_gate_direction_header.py`, 40 tests, hermetic)

Synthetic word tuples and in-test PDFs; no provider, no corpus, `_plan_native_table_first` called directly
(no `process()`, so the ambient-ollama traps do not apply); the lane tests need no `_available_engines_for_agentic`
patch beyond the one already used in the gh916 file.

- P7 difference pins: same words, only one line's direction changes (ships vs DEFER `foreign_direction`), and
  the gate switched off ships (the exact-pass it overrides exists). Direction not vocabulary: a foreign word
  not in the grid does not fire.
- Must-not-fire: single-direction table (all vertical); jitter 0, 1e-6, 0.01, 0.5, 2.0 degrees both signs;
  per-block membership (block 1 all vertical, block 2 all horizontal, disjoint vocabularies); a foreign word
  the grid does not carry may lack a key. Must-fire: 4, 45, 90, 180 degrees; a tie between two directions.
- Failure contract: omitted skips; extraction-failed, empty map, default-empty, plain `{}`, wrong type, missing
  key, no block/line indices, four unusable values; empty map with nothing carried; extraction failure
  returned not raised.
- Wiring: upright orchestrator (clean / same-direction / foreign stamp: events differ by exactly the gate event
  with `foreign_direction`); extraction failure at that site; the cell-repair re-plan carries a non-empty map and
  DEFERs on the stamp; rotated `attempt_rotated_native_table` (clean and same-direction stay on
  `ROTATED_SHIP_QUARANTINED`, the foreign stamp changes the reason to `ship_gate:foreign_direction`).
- H-V3: difference pin (band dropped from the grid defers, kept ships); no header row on the page; 2 lanes is
  not a header, 3 is; two words on one lane; a word between lanes; a row carrying a number; reach 5 pitches in,
  6 out; a row below the first data row.

## Measurement (page = one (doc, page); scripts in the session scratchpad, counts and basenames only)

Gate on this branch, the same loader as #920's benchmark (`attempt_rotated_native_table` for rotated, the census
`native_text` for upright). `socr.__file__` asserted inside the worktree. No page had a direction fault.

### 35 rotated pages (14 wrong / 20 cosmetic / 1 correct by fable.jsonl)

| measure | #920 (round 7) | this branch |
|---|---|---|
| wrong pages stopped | 12 / 14 | **14 / 14** |
| other pages stopped | 6 / 21 | 8 / 21 |
| `foreign_direction` fires | n/a | 4 pages |
| `header_band_missing` fires | n/a | 1 page |

- Wrong newly stopped: id 00 (Barrot p15, `foreign_direction`: running head "INPUT SPECIFICITY AND
  IDIOSYNCRATIC SHOCKS", 1557) and id 17 (Fama p561, `header_band_missing`: 12 words over 12 lanes, 13 absent).
  Id 32 (Martens p41) also fires `foreign_direction` (it was already stopped by `label_row_missing`).
- Others newly stopped: ids 31 and 33 (cosmetic), both `foreign_direction`. `header_band_missing` adds no
  other rotated page.

### 92 upright SHIP pages (census.jsonl, deduped by (doc, page))

| predicate | pages | of 92 |
|---|---|---|
| data_row_missing | 6 | |
| label_row_missing | 20 | |
| sign_detached / row_order / cell_order | 0 | |
| **foreign_direction** | 2 | |
| **header_band_missing** | 17 | |
| any | 40 | 43% (was 25) |

New fires versus #920's 25: **15 pages** (13 `header_band_missing` only, 2 `foreign_direction` only; 4 more
`header_band_missing` pages already fired `label_row_missing`: woodford 791, bybee 83, barry 19, hack 38). Ids,
renders and native markdown are in `~/.local/state/socr-housekeeping/gh917/newfires/` (`index.json`,
`<id>.png`, `<id>.md`). My reading of each (I looked at all 15 renders; `grep` of the missing words in the
page's non-table text for the 13 band fires, which is where the header words end up):

| id (doc_page) | predicate | reading |
|---|---|---|
| 2020__bybee_kelly_manela_28 | header_band | real: grid has an EMPTY header row and the values sit shifted; Mean / S.D. words missing from it |
| 2022__ayivodji_43 | header_band | real: `Models h=1..h=4` are plain text lines above the grid, columns unlabelled |
| 2023__bybee_79 | header_band | real: `SB Ctrl SB Ctrl` sub-header not in the grid |
| 2025__tabatabaei_61 | header_band | real: `Mean Median SD` x2 sub-header not in the grid |
| 2024__gong_li_zhang_53 | header_band | real: `Quintile ALL H M L` header not in the grid |
| 2025__piller_schranz_33 | header_band | real: spanning `All` / `Speeches` header words missing from the grid |
| 2023__lopez-lira_51 | header_band | real: `Intraday News` group header word missing |
| 2013__Phillips_Zhdanov_52 | header_band | real, minor: section heading `Ownership and Instruments` loses `Instruments` |
| 2023__bybee_81 | header_band | real, minor: panel label `B. Net MKT` not in the grid |
| 2022__ayivodji_38 | header_band | false: the row is the caption `Table 4: News Relative RMSE...` |
| 2022__ayivodji_41 | header_band | false: caption `Table 7: ...` |
| 2022__gholampour_52 | header_band | false: title `Table A7: Alternative Quantifications Summary` |
| 2024__eskildsen_67 | header_band | false-ish: panel caption `(b) Global ex-US` sits above the grid as text |
| 2024__gourier_39 | foreign_direction | real: a diagonal watermark ("For Peer Review Only / SFS Cavalcade NA 2025") is carried in cells |
| 2025__gurkova_81 | foreign_direction | false: the diagonal column headers are the table's own content (accepted false DEFER of rotated-header tables) |

So 4 of 13 band fires are captions or titles whose words sit one per lane (design predicted this) and 1 of 2
P7 fires is a legitimately diagonal header. Net: 15 new fires, about 10 real, about 5 false DEFER, each false
one costing one model read. These are my readings from the renders, not a vision audit.

## Mutation (copy of src + tests + pyproject in a temp dir, `socr.__file__` canary, uncapped `count == 1`)

24 of 24 killed against `tests/test_gh917_gate_direction_header.py`:

| mutant | failing tests |
|---|---|
| P7 call removed | 20 |
| header-band call removed | 3 |
| dir map without `TEXTFLAGS_WORDS` (image-block fixture) | 2 |
| extraction failure ignored | 2 |
| empty map ignored | 1 (the first version of the test survived: its "unrelated" vocabulary still carried letters of the fixture, now disjoint) |
| wrong type ignored | 2 |
| missing key ignored | 6 |
| unusable value treated as horizontal | 4 |
| extraction raises instead of returning a fault | 2 |
| tolerance zero (raw equality) | 3 |
| tolerance coarse (pi/2) | 10 |
| direction as axis, not vector | 2 |
| membership page-wide | 4 |
| H-V3 lane minimum 2 | 1 |
| H-V3 one-per-lane dropped | 1 |
| H-V3 off-lane word allowed | 1 |
| H-V3 reach unbounded | 1 |
| H-V3 rows below the first data row | 1 |
| H-V3 numeric row allowed | 1 |
| H-V3 present words still fire | 24 |
| upright site drops `line_dirs` | 2 |
| rotated site drops `line_dirs` | 1 |
| cell-repair site drops `line_dirs` | 1 |
| `plan_native_table` does not forward `line_dirs` | 20 |

## Results

Focused file: 40 passed. Full suite, default OLLAMA_HOST, in the background to a log: 6020 passed, 2 skipped,
4 xfailed, 0 failed (counted from the progress lines, since `-q -q` suppresses the summary; the box was loaded,
about 70 min wall clock). `uvx ruff@0.16.0 format --check .` clean.

## Follow-up

- The 5 false header-band DEFERs are captions and titles. A caption word list would be a magic vocabulary;
  the principled fix is #925 (the rowizer absorbs the band) with the gate staying as the net.
- Diagonal column headers (gurkova p81) defer on P7. If rotated-header tables matter, the carried words'
  direction could be compared against the header ROW's own direction; not done here (a missed foreign word can
  ship a wrong cell).
- The #918 quarantine is NOT lifted. With this PR 14 of 14 audited wrong rotated pages defer; the lift
  protocol (all 449 rotated pages re-run, vision audit of every SHIP) is the separate PR B.
- No STATUS.md / TICKETS.md entry: this is an issue-driven PR, not a plan-folder ticket.

## Round 2 (PR #926, Astra REJECT)

### 1. Blocker: the cell-repair path turned a gate DEFER into a REFUSE

`_repair_native_table_cells` re-plans the repaired grid with `plan_native_table`. It returned `None` for any
non-SHIP re-plan, so a DEFER (a gate fault, a P7 plumbing fault, any DEFER) lost its faults and the caller fell
through to `_refuse_native_table_first`: `native_table_structure_failed` / `native_table_unverifiable` set, the stale
cell-mismatch reason kept. The gate stopped being DEFER-only on that path, and the real reason was silent.

Fix: `_repair_native_table_cells` returns `(markdown, replan)` and keeps the re-plan. The repair now runs in the
`_phase_agentic` page loop (before the lane chain, for exactly the pages that would reach
`_apply_native_table_first`), so a DEFER can still reach `route_page`: the faults are recorded with
`_record_native_ship_gate` (same `native_ship_gate_deferred` event as the other sites), the page's plan is dropped
so it takes the whole-page route, and with no provider it gets the same native-text fallback an upright DEFER gets at
plan time. Only a non-DEFER failure (the transcription did not confirm, the splice failed, REFUSE/CELLS on re-plan, or an
unreadable text layer) still refuses, as before. `_apply_native_table_first` takes the already-resolved
`cell_repair`.

Tests (`TestCellRepairKeepsTheGateDeferOnly`, through `process()`, parametrised over provider / no provider, a
CELLS page whose first cell the grid got wrong and a repair that confirms it): the re-plan sees extraction failed,
empty map, missing keys (the three plumbing modes) and a foreign stamp (a real post-repair gate fault). For each: the repair
ran, exactly one gate event with the expected predicate, no `native_table_cell_unresolved`, no
`native_table_cell_repaired`, no refuse state, the page is routed once with a provider. Difference pins: the same page with
a clean repair SHIPs (no route, `native_table_cell_repaired`, no gate event), and a repair-time deferral matches a plan-time
deferral of the same stamped page (routes, status, refuse flags).
Mutants killed: faults not recorded, DEFER turned back into a refuse, re-plan discarded, re-plan run without directions.

### 2. `line_dirs` is required

`plan_native_table(..., *, line_dirs)` and `native_ship_gate(words, markdown, *, line_dirs)` have no default. `None` is
now a `direction_unavailable` DEFER, not a skip. The only way to skip P7 is `LineDirections.unchecked_for_tests()`
(grep-able; `unchecked=True`), used by the existing tests that do not exercise P7 (`UNCHECKED` in the gh916, native-table
and rotated test files). Tests: omitting the keyword is a `TypeError`; `None` DEFERs; the sentinel skips even with a foreign
word present. Mutants killed: `None` skips again; the sentinel ignored.

### 3. P7 tolerance: measured, not derived from width

The old tolerance `atan2(snap, max(extent, snap))` reached up to 45 degrees on narrow tables, and lane displacement is
not evidence of provenance. Replaced by `_SAME_TEXT_DIRECTION_TOL_RAD = 1e-5`.

Measurement (`jitter917.py` in the session scratchpad): for each of the 160 table blocks with a table geometry on the 35
rotated + 92 upright pages (124 pages; 3 pages have none), the maximum pairwise angle among the distinct direction vectors of the lines
that carry the block's CORE-row words (14913 line keys):

| max pairwise deviation within a block's core lines | blocks |
|---|---|
| exactly 0 (one direction vector) | 159 |
| > 0 up to 1e-3 rad | 0 |
| > 0.2 rad | 1 (Martens p41, 90 degrees: a genuine running head on a core row's y-band, i.e. foreign text, not same-table jitter) |

So the observed same-table deviation is 0; the tolerance is a float-noise floor. PyMuPDF `dir` comes from float32-precision
text matrices (about 1.2e-7); 1e-5 rad (0.0006 degrees) is about 100x that and about 17x below the smallest deliberate rotation
the tests treat as foreign (0.01 degrees = 1.7e-4 rad). Width plays no part. Tests: jitter up to ~1e-5 rad does not fire; 0.01,
0.5, 4, 10, 45, 90, 180 degrees fire; a narrow table (extent below the width at which the old formula tolerated 10 degrees) fires on a
10 and a 90 degree foreign line. Mutants killed: tolerance 0, 1.6 rad, 0.2 rad, width-derived again.

### 4. Header-band known limitations (not expanded; for #925's scope note)

`header_band_missing` is a net, and these are false negatives relative to an ideal check, not regressions versus main:
- Whole-word containment inside the data-derived x-range: a header wider than the carried words' bounding box (an edge header
  that overhangs the first or last data column) is outside `inside` and is not seen.
- Numeric headers (years, `(1)`, `(2)` column numbers): a row with any numeral is excluded by design, so a numbered header band
  is never flagged.
- Labels repeated elsewhere: `has_word` is substring presence per block, so a header word that also occurs in any cell of the grid
  counts as present even when the header row itself was dropped.
- Multiword headers that share a lane: two words over one lane fail the one-per-lane rule, so a header whose label is split across
  words on one lane is not flagged.
#925 (the rowizer absorbing the band) should cover these; the gate predicate stays the safety net, with these gaps.

### Re-measurement

Same loader and sets as round 1, after all round-2 changes (required `line_dirs`, measured tolerance, repair routing).
No page flips on either set (0 of 35 rotated, 0 of 92 upright), so the round-1 tables stand:

| set | measure | round 1 | round 2 |
|---|---|---|---|
| rotated (35) | wrong stopped | 14 / 14 | **14 / 14** |
| | others stopped | 8 / 21 | 8 / 21 |
| | `foreign_direction` / `header_band_missing` pages | 4 / 1 | 4 / 1 |
| upright SHIP (92) | pages firing (any) | 40 | 40 |
| | `foreign_direction` / `header_band_missing` | 2 / 17 | 2 / 17 |
| | new fires versus #920's 25 | 15 | 15 (same pages) |
| both | pages with a `direction_unavailable` fault | 0 | 0 |

The measured tolerance (1e-5 rad) changes no verdict because same-table deviation on the corpus is exactly 0 and every
foreign line is far above it (the smallest foreign angle on a firing page is tens of degrees).

### Mutation

31 of 31 killed against `tests/test_gh917_gate_direction_header.py` (canary on `socr.__file__`, uncapped `count == 1`). New or changed
this round: repair drops the faults (10 failing tests), DEFER turned back into a refuse (10), re-plan discarded (10), re-plan run
with the unchecked sentinel (10), `None` skips P7 (1), sentinel ignored (1), tolerance 0 (3), 1.6 rad (15), 0.2 rad (3),
width-derived again (2), `plan_native_table` forwarding the sentinel (30). The round-1 set still dies (the plumbing-site mutants
now surface as `TypeError`s or the same pins).

### Results

Full suite, default OLLAMA_HOST, detached to a log: 6031 passed, 2 skipped, 4 xfailed, 0 failed (578 s).
`uvx ruff@0.16.0 format --check .` clean.
