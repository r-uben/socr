# GH-934: line-level prose/caption predicate, single-walk stub-first header band (retry of #925)

Status: NOT MERGED. The branch met the 127-page census bar after revision 2, was pushed as PR #938, and was closed after
review (see "Outcome" at the end). The sections below are the record of what was built and measured.

Revision 2 (ruling: option 2, no waiver, `ship_gate.py` untouched): in `_header_band_ys`, when `_is_prose_like_row` rejects a LANE-SHAPED row
(`_is_lane_shaped_row`: 2+ words, every word snaps, 2+ distinct lanes) after the walk has left main's rule, the whole stub recovery is
discarded and only the rows main's own rule absorbed are returned. Census (same 127 pages): 3 pages change (Boukus 38, Fama 561, Gong 53, grids
byte-identical to revision 1); Ayivodji 43 is byte-identical to main again and DEFERs on `header_band_missing`. Caption/prose absorbed: 0.
Partial-header ships: 0. Revision 1 figures below are kept as the record; the revision 2 census is in "Revision 2 census".

Branch `fix/934-prose-line-predicate`, cut from origin/main 0eb6121 (ancestry verified). Design: `~/.local/state/socr-housekeeping/gh934/design.md`
(designer, audited by Fable as GO-WITH-CHANGES). The rejected attempt is #933 / `fix/925-header-band-stub` (9aa2c40); not reused except the
label-cell idea.

## Change (`src/socr/tables/reconstruct.py` only; `ship_gate.py` untouched)

- `_header_band_ys`: the ONE header-band walk. `_extend_scope_for_header` and `_prepend_header_band` both call it and no longer loop
  themselves, so the rejected attempt's failure (two walks, the second seeing a caption as its own nearest row) cannot recur.
  - Up to the point where main's rule stops (empty row, numeric token, or a word that snaps to no lane) it is main's walk, byte for byte.
  - From that point on EVERY row is tested (not only rows main's snap test rejects): `_stub_row_eligible` and not `_is_prose_like_row`.
- `_is_prose_like_row`: 2+ words and no gap wider than `ALIGNED_RUN_GAP_MAX_WORD_SPACES` (2.0, existing, no new constant) page word spaces,
  OR the row's median font size differs from the data rows' (exact equality). Word space is `_median_word_gap` of the frame the walk runs in
  (upright rowize frame on rotated pages).
- `_stub_row_eligible`: each word is a label-region word or snaps to a lane, at least one snaps (from #925).
- `_word_size_map(page)`: `(block, line, word) -> span size`, plumbed as `word_sizes` through `rowize_from_words`,
  `rowize_from_words_chart_aware`, `_reconstruct_table_regions_for_words` -> `rowize_from_word_list` -> `_rowize_word_group` ->
  `_prepend_header_band` / `_extend_scope_for_header`. The key survives `_rotate_word_bbox`. No size map (bare `rowize_from_word_list(words)`
  callers, unreadable spans) makes the size clause inert, i.e. the single-run clause alone.
- Stub words of a recovered header go to the header row's label cell (`_is_label_region_word`).

## Tests

Full suite (default OLLAMA_HOST, nohup, one complete run): revision 1 6070 passed, 2 skipped, 4 xfailed; revision 2 6070 passed, 2 skipped, 4 xfailed in 902s. `uvx ruff@0.16.0 format --check .` clean.

`tests/test_gh934_prose_line_predicate.py`, 14 tests (revision 2: the spanning-header "known loss" test is now a fallback-to-main
difference pin, and the all-snap-caption and extend-site expectations are main's behaviour), all difference pins on synthetic geometry (the same page rowized twice, one thing changed):
stub-first header recovered (vs stub exemption off); label-region caption/prose line above the band rejected (vs predicate off); all-snap
caption above the stub header rejected (the Kalemli shape; needs the "every row past main's stop" placement); wide-sentence-space footnote
rejected by the size clause only; plain header and main-absorbed single-run rows unchanged (controls); label-only row not absorbed;
both sites (`_extend_scope_for_header` and the prepend site) pinned; Fable (iv): a ONE-word caption in the data font size IS absorbed
(known exposure, pinned); a one-run spanning header above a stub header is lost (known loss, pinned).

### Mutations (isolated copy of src+tests+pyproject under `~/.local/state/socr-housekeeping/gh934/mut/`, `socr.__file__` canary test, anchor count == 1 uncapped)

| mutation | killed by |
|---|---|
| control (none) | 15 passed |
| M1 drop single-run clause | 4 tests |
| M2 drop size clause | 1 test (the size-clause test only) |
| M3 drop reach (predicate only on rows main rejects) | 3 tests (all-snap caption, spanning-header, extend site) |
| M4 `n_words >= 2` -> `>= 1` | 1 test (the one-word-caption exposure) |
| M5 drop lane-word requirement | 1 test (label-only row) |
| M6 drop stub label cell | 7 tests |
| M7 no word space at the extend site | 1 test (extend site) |
| M8 (revision 2) drop the lane-shaped fallback clause | 3 tests (spanning-header fallback, all-snap caption, extend site) |

Revision 2 numbers: the test file has 14 tests (15 collected with the mutation-copy canary); the control above reads 15 for that reason.

Mutations also shown load-bearing on real input (census mutants, 16 gh925 pages, `~/.local/state/socr-housekeeping/gh934/census/m1..m3`):
M1 absorbs extra rows on 7 pages (Fama 753, Boukus 38/39, Ayivodji 43, Bybee 78, Hansen 28, Eskildsen 70); M2 absorbs the
Phillips-Zhdanov 52 footnote; M3 absorbs a row on Ayivodji 43.

## Census (main 0eb6121 vs branch, frozen sources, `socr.__file__` canary asserted; same 127 pages as #925)

Artefacts: `~/.local/state/socr-housekeeping/gh934/census/` (main/, branch/ frozen copies; `main.json` is byte-identical to gh925's `main.json`),
`~/.local/state/socr-housekeeping/gh934/changed/` (per-page before/after markdown + render + index.json). I viewed all four renders.

127 pages: 4 change (4 grids, 0 verdict-only). `header_band_missing` fires 18 -> 15.

| id | page | class | gate predicates main -> branch | plan action | numerics |
|---|---|---|---|---|---|
| 000 | Boukus 2006 p38 | header recovered (Panel A and B headers move into their tables; captions stay prose) | none -> none | n/a | identical, all tokens |
| 001 | Fama 2017 p561 (rotated) | header recovered (Country, rho1..12, Autocorrelations, Mean, Std. Dev.); still not a ship | header_band_missing + text_in_numeric_column -> text_in_numeric_column | defer -> refuse | removed: none; added: the digits 1..12 of the rho1..rho12 header labels (header text that main dropped entirely) |
| 002 | Ayivodji 2022 p43 | PARTIAL HEADER | header_band_missing -> none | n/a (upright) | identical, all tokens |
| 003 | Gong 2024 p53 | header recovered (two tables) | header_band_missing -> none | n/a | identical, all tokens |

- (a) The multiset of numeric tokens over ALL tokens: nothing removed on any page; identical on 000/002/003. On 001 the only change is added
  digit tokens from the `rho1..rho12` header labels (the regex picks the subscript out of the label). No data value moves.
- Caption or prose absorbed: 0 pages. Every caption and panel title next to these tables (Table 8 caption, "C:/D: Quintile-based Testing",
  "Panel A/B", "TABLE 1." title) stays outside the grid.
- The 7 pages where main already ships a caption inside header rows (Eskildsen 67 and 70, Kalemli-Ozcan 80, Gomez-Cram 8 and 10,
  Perico-Ortiz 41, Meyer-Wesseler 86) are unchanged (byte-identical to main). Out of scope. **Follow-up ticket needed:** main absorbs captions
  into header rows on those pages; the fix is a predicate applied to rows main's own walk absorbs (design V-all), which also loses headers
  where an in-table panel title sits between header and data.

### Partial-header ships (Fable iii) and the merge bar

**Ayivodji 2022 p43 (Table 8): DEFER on main (`header_band_missing`), SHIPS on the branch with no predicate firing.** The two-word spanning row
"Relative RMSE" is rejected as a single run; the lower row "Models | h=1..h=4" is kept. The spanning words stay in the page prose above the
table, no value is lost, but the grouping header is no longer in the grid and nothing in the gate notices. By the hard rule this is a NEW
SILENT SHIP and blocks merge unless caught another way; it is not caught (predicates empty, verified from the census).
This is the only such page in the 127. It is exactly the 1 of 23 band header the design predicted, and it is a spanning header.

Narrowest fixes (none applied; `ship_gate.py` is out of my ownership, #932 is editing it):
1. Gate-side: keep `header_band_missing` firing when a non-numeric, lane-shaped row (2+ words over 2+ lanes) sits directly above the first header
   row of a shipped grid. Needs the gate to see a row the rowizer rejected. Owner: #932 / ship_gate.
2. Rowizer-side (my file): when `_is_prose_like_row` rejects a row that is lane-shaped (every word snaps) directly after a stub-recovered
   row, discard the whole stub recovery and fall back to main's (empty) band. Ayivodji 43 returns to main's DEFER; every other recovery is
   unaffected on the census (the rejected rows there are label-region captions, not all-snap rows). Not measured; run the census before landing.
3. Accept: the spanning label survives as prose. Contradicts the hard rule; only the orchestrator can waive it.

### Revision 2 census (current)

Frozen `census/branch2/` vs `census/main/`, `socr.__file__` canary asserted; artefacts `~/.local/state/socr-housekeeping/gh934/changed2/`.
127 pages: 3 change, 0 verdict-only; `header_band_missing` fires 18 -> 16.

| page | class | predicates main -> branch | plan action |
|---|---|---|---|
| Boukus 2006 p38 | header recovered | none -> none | n/a |
| Fama 2017 p561 | header recovered, still not a ship | header_band_missing + text_in_numeric_column -> text_in_numeric_column | defer -> refuse |
| Gong 2024 p53 | header recovered | header_band_missing -> none | n/a |
| Ayivodji 2022 p43 | unchanged (byte-identical to main, DEFER on header_band_missing) | - | - |

All three grids are byte-identical to the ones I viewed in revision 1 (their renders were viewed then; the census page and grid are the same). Numerics: nothing
removed on any page; identical over all tokens on Boukus 38 and Gong 53; on Fama 561 only the digits 1..12 of the rho1..rho12 header labels are added.
Caption/prose absorbed: 0. Partial-header ships: 0. Boukus 38, Fama 561 and Gong 53 keep their recoveries.
Cost: header recovery is not attempted on a page whose stub header has a lane-shaped spanning row above it (Ayivodji 43 shape) or an all-snap
caption above it (the Kalemli shape); those pages behave as on main.

### Revision 1 notes: expected header loss (Fable ii)

About 9%: pooled 11 of 121 multi-word header rows score <= 2.0 word spaces, all group-spanning headers. The earlier "1 of 23" is the stub-band
sample only. On the 127-page census exactly one row is lost (Ayivodji 43). A lost spanning row above a kept header is the partial-header shape.

### Size clause (Fable i)

The size clause rescues two prose lines on the design fixture (Phillips-Zhdanov 52, a sentence space of 2.28 word spaces, and one Fama 728
line); the result depends on walk order, because the walk stops at the first rejected row and a row below an already-rejected one is never
tested. On this census the size clause alone changes exactly one page (Phillips-Zhdanov 52, mutant M2); dropping either clause alone does not move the Fama 728 line (it is rejected by both, or not reached in this walk order). Every measured stub-band header shares the data font size
exactly; in-segment headers do not (6 of 111), so the size clause is only safe at the band.

### One-word caption exposure (Fable iv)

A one-word caption in the data font size has no gap to measure and the same size, so neither clause rejects it; it is absorbed as a header cell
if it snaps to a lane. Pinned by `test_one_word_caption_in_the_data_font_size_is_absorbed`. Not seen on the census (zero caption absorptions); the
fixture's one-word band rows were all headers.

## Not done / follow-ups

- (Decided: option 2, implemented in revision 2.)
- Follow-up ticket (to be filed by the coordinator) for the 7 main-absorbed caption pages (V-all).
- #921 page-level prose gate (separate predicate, design section 6).
- No `STATUS.md` / `TICKETS.md` entry: GH-934 is not in a plan folder.

## Outcome: rowizer change NOT merged (PR #938 closed, 2026-10-01)

The branch `fix/934-prose-line-predicate` (b0ee21d) met the census bar: on 127 pages, 0 captions absorbed and 0 partial
headers. Astra's review still found geometries the census did not contain that turn a main DEFER into a silent wrong
SHIP:
- a one-word caption or note ("Notes") that snaps to a lane above a stub header;
- a same-size caption with a wide gap, or a roman-numbered caption;
- a one-word spanning heading in a larger font: the size clause rejects it, but the lane-shape fallback needs 2+ words,
  so the stub recovery stays and a partial header ships;
- the size map is not plumbed through `born_digital.py`'s `rowize_from_word_list` call, so the size clause fails open
  there;
- `_extend_scope_for_header` measures word space before the upright rotation is applied.

**Why it was dropped rather than patched again:**
- The gain is small and cheap to forgo. In revision 2, 1 page of 127 (Gong 53) skips one model read, Boukus 38 gets a
  better header, and Fama 561 still refuses.
- The risk is in the class this repo ranks worst: a page that DEFERs on main ships a wrong header with no gate signal.
- On the census, every stub-first band main drops fires `header_band_missing` (18 pages), so that miss costs one model
  read today. `header_band_missing` does not detect every missing header: a missing one-word spanning heading (the
  #938 counterexample) is not something it sees, which is why a partial band must never be shipped as if complete.

**Where the measurement goes instead:**
- The prose-line predicate (single run at `ALIGNED_RUN_GAP_MAX_WORD_SPACES`, plus a font-size clause) belongs on the
  DEFER side of the ship gate, not in the rowizer.
- #936: flag a SHIP whose header rows are prose-like (main already ships captions inside headers on 7 pages).
- #921: flag a non-table page; the design measured 11/11 caught with 3/82 real tables deferred.
- A false DEFER there costs one model read, so the asymmetric cost works in its favour.
