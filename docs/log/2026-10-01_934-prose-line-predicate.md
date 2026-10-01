# GH-934: line-level prose/caption predicate, single-walk stub-first header band (retry of #925)

Status: IMPLEMENTED, MERGE BAR NOT MET. One partial-header ship (Ayivodji 43) is a new silent ship under the
hard rule. Do not merge until it is decided (see "Merge bar"). Not pushed.

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

Full suite (default OLLAMA_HOST, one complete run): 6070 passed, 2 skipped, 4 xfailed in 3478s. `uvx ruff@0.16.0 format --check .` clean.

`tests/test_gh934_prose_line_predicate.py`, 15 tests, all difference pins on synthetic geometry (the same page rowized twice, one thing changed):
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

### Expected header loss (Fable ii)

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

- Decision on Ayivodji 43 (fix 1, 2 or 3 above).
- Follow-up ticket for the 7 main-absorbed caption pages (V-all).
- #921 page-level prose gate (separate predicate, design section 6).
- No `STATUS.md` / `TICKETS.md` entry: GH-934 is not in a plan folder.
