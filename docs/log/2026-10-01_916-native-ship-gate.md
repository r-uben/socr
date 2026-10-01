# GH-916: a ship gate in front of native-first SHIP (2026-10-01)

Branch `fix/916-native-ship-gate` from origin/main ce59b40 (ancestry verified).

## What changed

`plan_native_table` ships on `EXACT_PASS`, which pairs rows by numeric multiset and
ignores standalone sign glyphs. New `src/socr/tables/ship_gate.py` runs words-only checks
on the shipped markdown after an `EXACT_PASS`, before returning SHIP. Settled decisions
(Fable + Astra) applied:

1. **Every predicate DEFERs; none REFUSEs.** The plan is `DEFER` with reason
   `ship_gate:<predicates>` and `NativeTablePlan.faults`. The page goes to `route_page`
   + judges + table ladder (a REFUSE would send an upright page to the D3 image floor with
   no model attempt).
2. **Fault carried forward as ONE new audit kind**, `native_ship_gate_deferred` (no existing
   family fits: `rotated_native_table_quarantined` is rotated-only and means a different
   thing). Emit sites: `_plan_native_table_first` upright branch and rotated branch, both
   through `_record_native_ship_gate`. Added to `resume_restore_kinds()`. Flush/restore
   count test in `tests/test_gh916_native_ship_gate.py` (1 emitted, 1 restored).
3. **Rotated quarantine (#918) stays.** The gate runs inside `plan_native_table`, so a
   rotated page it defers takes the gate reason; a rotated page it passes still hits
   `ROTATED_SHIP_QUARANTINED`.
4. Astra's corrections:
   - `detached_sign_pairs(words)` (in `reconstruct.py`) is the one words-only statement of the
     #887 contact criterion. `_reattach_detached_signs` (the `find_tables` cell merge) and P2
     both call it; `_flush_on_the_left` was lifted to module level for it. #887 tests pass
     unchanged.
   - P2 also catches a sign at the END of a populated cell, and binds the contact to the
     output row: only source lines whose numeric multiset contains the row's tokens count; a
     contact on a different line with the same value is not evidence.
   - P5a table membership is source-side: lanes are the x-clusters of numeric words of rows that
     pair UNIQUELY (equal multiset, unique on both sides). A source row is a table row if it
     sits between the first and last paired row with >= 2 distinct lanes, or anywhere with
     >= max(modal paired width, `_MIN_LANES_PER_ROW`) lanes. Fewer than two paired rows ->
     abstain. The output-derived y-band scope (`_effective_native_rows_for_output`) is not used.
   - P5c restricted to the table's own x-extent (bounding box of the paired rows' words that the
     grid carries, so another column's words never widen it) and to rows strictly between the
     first and last paired row. Words are compared after NFKC (ligatures).
   - P1 order checks use only unique exact-multiset pairs (no ordinal fallback) and abstain
     when a multiset repeats on either side. Checked per table block.

Tolerances are existing named rowizer quantities (`_LANE_X_TOL_PT * _LANE_SNAP_MULT`,
`_MIN_LANES_PER_ROW`). No new threshold.

Predicates implemented: `sign_detached` (P2), `row_order` + `cell_order` (P1),
`data_row_missing` (P5a), `label_row_missing` (P5c). Not implemented: P7 (foreign text
direction) needs `get_text("dict")` line direction, which `plan_native_table(words, markdown)`
does not receive; it was not in the settled list. P5b (narrow missing row) abstains by
construction (a row narrower than the modal width outside the paired span is not claimed as table
membership). P3 stays withdrawn.

## Measurement

### Rotated set (35 pages in gh902/q917, gate run directly, quarantine bypassed)

Per-predicate firing (pages): `label_row_missing` 9 on the 14 wrong + 7 on the 21 others,
`data_row_missing` 2 wrong + 0 others, `sign_detached` 2 wrong + 0 others, `row_order` 0,
`cell_order` 0.

- Wrong pages stopped: **12 / 14**. Not stopped: 00 (Barrot p15, running head in cells: needs P7
  direction) and 17 (Fama p561, dropped header band: the design's accepted rowizer residual).
  Page 32 (Martens p41) is stopped, by P5c rather than P7.
- Others stopped: **7 / 21** (ids 09, 11, 14, 21, 23, 26, 31, all P5c; the reported missing words
  are real grid content, e.g. formula fragments, "First", "T.", "(Continued)").
- `sign_detached` fires on exactly 12 (Fama p427) and 20 (Fama p571), the two sign pages.

### Upright SHIP pages (census.jsonl at run time: 90 lines, 64 upright SHIP pages in 33 docs)

| predicate | pages firing | of 64 |
|---|---|---|
| sign_detached | 0 | 0% |
| row_order | 0 | 0% |
| cell_order | 0 | 0% |
| data_row_missing | 6 | 9% |
| label_row_missing | 11 | 17% |
| any | 16 | 25% |

**This is not a small fraction; reviewer decision needed on P5a/P5c before the PR.**
The round-1 firing ids are superseded by the Round 2 list below (basenames and pages are recorded
in this log; a local json beside the census is supplementary only).
I looked at the page image for 13 of the 16 pages (render next to the grid). Every one was a real grid defect or
not a table at all: woodford 787 and ljungvist 7 (an index page and a table of contents shipped
as tables), bybee 67 (a numbered prose list), brochet 21 (the two "Dependent Variables" data rows
and the header are missing from the grid), segal 66 (Adj.-R2 and Obs rows missing),
ayivodji 38 and 41, bybee_kelly 36, hansen 28 (the table's Note paragraph was absorbed into the
grid as rows and words dropped), bybee 83 (panel B-D label rows dropped), ramey 104 (multi-line
text cells; the numeric rows have no grid row). One false fire was found and fixed during the
measurement: cieslak 63 (typographic ligature "Staff" in the PDF vs "Staff" in the grid; fixed by NFKC).
Viewed but not diagnosed against the grid text: faust 44 (a tilde-r symbol label is missing from
the grid; looks real, could be called cosmetic), levy 105 (the word is in the intro paragraph above
the table; whether the grid swallowed it was not checked). Not viewed: woodford 791 and 802 (same
index document as 787), bybee 78.
So the predicates mostly fire on shipped grids that are wrong, but 25% is a large yield cost:
each fired page loses the native SHIP and takes a model read. The ship-or-not call on P5c in
particular (17%) is the reviewer's.

## Tests (`tests/test_gh916_native_ship_gate.py`, hermetic: synthetic word tuples, one synthetic PDF)

Difference pins per predicate (fault present -> DEFER with predicate; gate off -> SHIP, proving the
exact-pass it overrides; fault absent -> SHIP): sign (bare cell, end of populated cell, repeated
value bound to the right output row), row order, cell order, dropped data row (first and last row;
an interior drop is already refused by the verifier's own row count), dropped panel label. Must-not-fire:
placeholder dash a column gap from its number, range `1990 - 2000` (flush both sides), right-column
prose numbers beside the table, two lane-aligned prose numbers below the table, caption below the
last row, other-column prose between rows, identical rows (ambiguity abstain).
`process()` lane test, parametrised over provider / no provider, pins the DIFFERENCE between the
clean and the row-dropped page (exact-pass event vs `native_ship_gate_deferred`, `route_page` called
once vs not, no D3 cell-unresolved marker), not an absolute status tuple. Flush/restore count test and
allowlist membership test.

## Mutation (copy of src + tests + pyproject in /tmp/socr-mut-916-*, canary on `socr.__file__`,
uncapped `anchor.count == 1`)

Each mutant is run against `tests/test_gh916_native_ship_gate.py` (plus the #887 suite for the shared
helper). All were killed, canary passed:

| mutant | failing tests |
|---|---|
| `sign_detached` (`pairs = []`) | 3 (bare cell, end of populated cell, repeated-value binding) |
| `row_order` (`if False`) | 1 |
| `cell_order` (`if False`) | 1 |
| `data_row_missing` (`member = False`) | 4 (unit + both lane-provider cases + resume) |
| `label_row_missing` (`absent = []`) | 1 |
| gate not called in `plan_native_table` | 9 |
| `detached_sign_pairs` returns nothing | 3 in the gate file + 5 in `test_gh887_reattach_detached_signs.py` (one helper, both consumers) |
| `_flush_on_the_left` always False (range guard) | 1 (the `1990 - 2000` must-not-fire test) |
| event not recorded in `_record_native_ship_gate` | 3 |
| kind removed from `resume_restore_kinds()` | 2 |

## Results

Focused: `tests/test_gh916_native_ship_gate.py` 18 passed. Full suite, default OLLAMA_HOST, `-v` to a log: **5937 passed, 2 skipped,
4 xfailed** (3247 s; the box was loaded by an unrelated ollama job, which made some tests take ~100 s).
The #917 log recorded 5917 passed; I did not re-measure that baseline on this tree.
`uvx ruff@0.16.0 format --check .` clean.

## Follow-up

- Decide P5c (17% upright) and P5a (9%): each fired page was a real defect, but the yield cost is
  real. Both are separate functions in `ship_gate.py` and can be dropped from `native_ship_gate()`
  independently.
- Lifting the #918 rotated quarantine is NOT done here. With this gate 12/14 wrong rotated pages are
  stopped; 00 and 17 would still ship and are the known residuals (running heads; dropped header band).
- Several fires are rowizer defects (Note paragraphs absorbed as rows; index/TOC pages emitted as
  tables); the gate defers them, it does not fix them.

## Round 2 (PR #920 review: Astra ACCEPT-WITH-FIXES, cubic, Fable's 7 new fires)

Full census, deduped by (doc, page): 92 upright SHIP pages. Round 1 fired on 23 of them (25%).

### Changes

1. `data_row_missing`: membership is now spatial. Table lanes and the span come from CORE paired
   rows (>= 2 numeric words in >= 2 lanes that >= 2 paired rows share), so a prose line paired by one
   stray number no longer stretches the span. A candidate row must lie inside the span extended
   outward by rows no further apart than `max(_PANEL_GAP_ROWS * pitch, _SPLIT_GAP_MIN_PT, the
   gap between the table's own blocks)` that have as many lanes as the table's modal row. A numeric
   full-width line far above or below is not a member. `_PANEL_GAP_ROWS = 2` is a named, documented
   constant (one label or blank row between two data blocks of one table); the other terms are
   derived. The header-subset exemption is gone. Presence is counted over the grid's numeric tokens:
   a candidate row is present only if all its numbers are still unclaimed, and claiming uses them up
   (a dropped copy of a repeated row now fires; numbers merged into another row's cells do not).
2. `sign_detached`: bound by row AND column. The output row's numeric sequence must equal the x-sorted
   numeric words of every candidate source line, and the contact must be on the word at the number's
   own position; any candidate line without that contact (a placeholder row with identical numbers)
   or no matching line means abstain. The sign may end a bare cell, a populated cell, or be attached
   as a tail (`label-`). Leading decimals (`.23`) pair: `starts_a_number` is shared by
   `detached_sign_pairs` and `_reattach_detached_signs`, and the gate's source-side numeric test accepts
   `.23` (the verifier's output side already reads it as a number; its source regex does not).
3. `label_row_missing`: per-cell comparison (no match across cells), NFKC, hyphenated line breaks
   joined (`Evalu-` + `ation`), repeated labels counted (each occurrence in the grid is used once).
   Not counted as missing: a row that starts with Note/Notes/Source (nor anything below it), and a
   prose-like row, defined from this table's own lanes (more words than lanes and spanning first to
   last lane). Rows are only those inside the CORE span, so a Notes paragraph swallowed into the grid
   no longer extends it.
4. Tests added: rotated gate-event (same rotated page with and without a dropped row: quarantine
   event alone vs gate event), gate error (a raising predicate defers, never ships, is recorded),
   gate DEFER vs ordinary DEFER through `process()` (provider and no provider): audit events differ
   by exactly the gate kind, routing, markdown and every sidecar field except per-run checksum and
   clocks are identical. Counterexample / must-fire tests for each change above. 36 tests in the file.

### Re-measurement (counts per predicate; page = one (doc, page))

| set | measure | round 1 | round 2 |
|---|---|---|---|
| upright SHIP (92) | pages firing | 23 | 22 |
| | data_row_missing | 7 | 7 |
| | label_row_missing | 17 | 16 |
| | sign_detached / row_order / cell_order | 0 / 0 / 0 | 0 / 0 / 0 |
| rotated (35) | wrong pages stopped | 12 / 14 | 12 / 14 |
| | others stopped | 7 / 21 | 6 / 21 |
| | label_row_missing wrong / other | 9 / 7 | 8 / 6 |
| | data_row_missing wrong | 2 | 2 |
| | sign_detached wrong | 2 | 2 |

Fable's 7 new fires (verdicts from `new7`): the two false ones (gomez-cram p10, piller p33) no
longer fire. Of the five true ones, hack p38, wang p11, bugel p11 and jiang p44 still fire.
fernandez-fuertes p73 no longer fires: its flagged lines are the Notes paragraph below the table (what
the geometry now treats as prose), and Fable's "true" there rests on a note line that is not recovered
elsewhere, which the gate cannot see. That is one lost true positive, the price of the Notes rule.
Net on the upright set: 9 pages stopped firing (the two false ones, the Notes/body-paragraph ones,
levy 105, woodford 791) and 8 pages newly fire, all `label_row_missing` (boukus 46, gow 48, bybee 10,
cook 21, hansen 29 and 30, barry 19, jiang 50). I did not look at those 8 page images; the reported
words are row or panel labels absent from the grid ("United States" heading x4, "developed countries",
"Panel", "Introductory Statement", "Convenience", a "***" stars-only line), so they read as real
label losses, but that is unverified.
The rate is still about one page in four. Rule on P5c (16 of 92) and P5a (7 of 92) is open.

### Pages that fire (upright SHIP, round 2): basename, page, predicates

| basename | page | predicates |
|---|---|---|
| 2003__woodford.pdf | 787 | data_row_missing |
| 2003__woodford.pdf | 802 | data_row_missing |
| 2006__boukus_rosenber__information_content_fomc_minutes__WP.pdf | 46 | label_row_missing |
| 2008__faust_wright__efficient_prediction_of_excess_returns.pdf | 44 | label_row_missing |
| 2016__ramey__shocks.pdf | 104 | data_row_missing, label_row_missing |
| 2018__brochet_kolev_lerman__information_transfer_conference_calls__RAS.pdf | 21 | data_row_missing |
| 2018__ljungvist_sargent__macro.pdf | 7 | data_row_missing |
| 2021__gow_larcker_zakolyukina__non_answers_during_conference_calls__JAR.pdf | 48 | label_row_missing |
| 2023__bybee__the_ghost_in_the_machine_beliefs_with_llm__WP.pdf | 10 | label_row_missing |
| 2023__bybee__the_ghost_in_the_machine_beliefs_with_llm__WP.pdf | 67 | label_row_missing |
| 2023__bybee__the_ghost_in_the_machine_beliefs_with_llm__WP.pdf | 78 | label_row_missing |
| 2023__bybee__the_ghost_in_the_machine_beliefs_with_llm__WP.pdf | 83 | label_row_missing |
| 2023__cook_kazinnik_hansen_mcadam__local_language_models_financial_earnings_calls.pdf | 21 | label_row_missing |
| 2023__hansen_kazinnik__fedspeak_decipher__WP.pdf | 29 | label_row_missing |
| 2023__hansen_kazinnik__fedspeak_decipher__WP.pdf | 30 | label_row_missing |
| 2023__segal.pdf | 66 | data_row_missing |
| 2025__barry_bruns_kandemir_klose_smirnov_tillmann__emotions_monetary_policy__WP.pdf | 19 | label_row_missing |
| 2025__hack_istrefi_meier__systematic_origins_of_monetary_policy_shocks__WP.pdf | 38 | label_row_missing |
| 2025__wang_liu_chen__current_stance_vs_future_guidance_llm_evidence_on_how_pbc_communication_shapes_the_yield_curve__EL.pdf | 11 | label_row_missing |
| 2026__bugel_hidalgo_luetticke__unconventional_unified_narrative_mp_shocks__WP.pdf | 11 | data_row_missing |
| 2026__jiang_krishnamurthy_lustig_richmond__dollar_erosion_loss_of_reserve_currency_status__WP.pdf | 44 | label_row_missing |
| 2026__jiang_krishnamurthy_lustig_richmond__dollar_erosion_loss_of_reserve_currency_status__WP.pdf | 50 | label_row_missing |

### Mutation (copy of src + tests + pyproject, canary on `socr.__file__`, uncapped `count == 1`)

All 17 killed, one test class each unless noted: sign ambiguity all->any (1), sign column to any word
on the line (1), tail sign not seen (1), `starts_a_number` without leading decimal (1, plus #887 suite
passes), core needs 2 numerics and 2 lanes (1; the two conditions are redundant with each other, so
the mutant removes both), unbounded reach above (1), unbounded reach below (1), reach zero / no
extension (8), pool not consumed (2), prose-like off (1), Notes opener never matches (1), no
dehyphenation (1), cells concatenated (1), labels not counted (1), gate error propagates (2), rotated
emit site removed (1), upright emit site removed (5).

### Results

Focused file: 36 passed. Full suite: 5955 passed, 2 skipped, 4 xfailed (default OLLAMA_HOST, 1028 s). `uvx ruff@0.16.0 format --check .` clean.
