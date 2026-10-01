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

## Round 3 (Astra rejected round 2: false negatives)

### Changes

1. **Notes cut-off removed.** The `break` on a Notes/Source opener skipped every later label in the
   block. Label rows are only ever scanned strictly between the first and last CORE paired row, so a
   Notes/Source heading there is a table label and checking resumes after it. Rows below the last core
   row (a Notes paragraph swallowed into the grid) are never scanned, which is the note region. Test
   flipped: "Panel C" vanishing after "Notes:" with data resuming now FIRES.
2. **`_prose_like` removed.** Width plus word count classified a 5-word heading spanning 4 lanes as
   prose. Inside the span nothing is exempt. Counterexample test: a full-width genuine heading dropped
   from the grid fires. Prose is instead kept out of the CORE set by positive evidence (below).
3. **Outward scan.** `prev` no longer advances on every row. Only a full-width numeric row extends the
   span, measured from the current edge, so prose in between cannot bridge to a distant numeric line.
   Must-not-fire test: 11 prose rows at the table's pitch lead to a numeric line 12 pitches out.
4. **Shared rowizer change kept and pinned.** `starts_a_number` (leading decimal) stays in
   `_reattach_detached_signs`, now with direct output tests in `test_gh887_reattach_detached_signs.py`:
   leading-decimal merge WITH contact, the same shape without contact (unchanged), a tail sign with a gap
   before it (merged), a tail sign flush against its label (a hyphen, unchanged). Mutation: reverting only
   the merge condition fails 2 of them.
5. **`_PANEL_GAP_ROWS` derived.** On the 35 rotated and 92 upright pages: 1822 gaps between consecutive
   core paired rows, each divided by its block's median gap: median 1.0, p90 2.0, p95 2.25, p99 4.83,
   max 28.45 (the tail is index pages of doc woodford, which are not tables; counts above 2, 3, 4
   pitches: 131, 42, 21). The constant is 5 (p99 rounded up), was 2. Test at the limit: a full-width row
   exactly 5 pitches out is in, 6 is out.
6. **Two more core rules (needed to keep the Fable false fire quiet once the Notes rule was gone).**
   A paired row is core only if its non-numeric word count in the lane region is within this table's own
   median plus one word per lane (a Note line carrying numerals is prose, not a row). A source word whose
   number ends in `, . ; :` ("for 7, 5,") is not numeric for the gate. Tests for both.

### Re-measurement

| set | measure | round 2 | round 3 |
|---|---|---|---|
| upright SHIP (92 pages) | pages firing | 22 | 20 |
| | data_row_missing | 7 | 4 |
| | label_row_missing | 16 | 17 |
| | sign_detached / row_order / cell_order | 0 | 0 |
| rotated (35 pages) | wrong pages stopped | 12 / 14 | 12 / 14 |
| | other pages stopped | 6 / 21 | 4 / 21 |
| | label_row_missing wrong / other | 8 / 6 | 8 / 4 |
| | data_row_missing wrong | 2 | 2 |
| | sign_detached wrong | 2 | 2 |

Pages whose upright verdict flipped (round 2 to round 3):
- Stopped firing: woodford 787 and 802 (index pages emitted as tables; they fired only on
  section-number-like rows "3.1.", now excluded as sentence-punctuated numerals), ljungvist 7 (table of
  contents, same), bybee 67 (numbered prose list). These four are non-tables that still ship as tables;
  the gate no longer defers them. Say so rather than count it as a clean result.
- Started firing: cieslak 63 (header words fragmented across cells, the "Staff Rev." lines that round 2 had
  treated as prose) and theodoridis 1203 (not looked at).
- new7 (Fable): the two false fires (gomez-cram p10, piller p33) stay quiet. True: hack p38, wang p11,
  bugel p11, jiang p44 still fire; fernandez-fuertes p73 does not fire (Notes below the table; unchanged
  from round 2, a known lost true positive).
- new8 (all 8 true per Fable): all 8 still fire.
- Rotated flips, against round 1: pages 11, 23 and 31 (cosmetic, not wrong) no longer fire (two of the three already stopped in round 2); the 12 of 14 wrong pages are
  the same set (0 and 17 are the residuals).

### Pages that fire (upright SHIP, round 3): basename, page, predicates

| basename | page | predicates |
|---|---|---|
| 2006__boukus_rosenber__information_content_fomc_minutes__WP.pdf | 46 | label_row_missing |
| 2008__faust_wright__efficient_prediction_of_excess_returns.pdf | 44 | label_row_missing |
| 2016__ramey__shocks.pdf | 104 | data_row_missing, label_row_missing |
| 2018__brochet_kolev_lerman__information_transfer_conference_calls__RAS.pdf | 21 | data_row_missing |
| 2020__cieslak_vissing-jorgensen__the_economics_of_fed_put__WP.pdf | 63 | label_row_missing |
| 2021__gow_larcker_zakolyukina__non_answers_during_conference_calls__JAR.pdf | 48 | label_row_missing |
| 2023__bybee__the_ghost_in_the_machine_beliefs_with_llm__WP.pdf | 10 | label_row_missing |
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
| 2026__theodoridis__machine_learning_classics_to_deep_networks_transformers_diffusion__academic_press.pdf | 1203 | label_row_missing |

### Mutation

All killed, canary on `socr.__file__`, uncapped `count == 1`: prior set unchanged, plus new:
reach walks only on full-width rows (prose-bridging mutant, 1), Notes opener hides the rest (1), width
makes prose (1), constant 2 instead of 5 (1; the first version of this test read the constant from the
module and the mutant survived, it now hard-codes 5), prose-with-numerals is core (1), sentence-punctuated
numeral counts (1), rowizer merge condition reverted (2 in the #887 file).

### Results

Focused: 40 tests in the gate file, 15 in the #887 file. Full suite: 5963 passed, 2 skipped, 4 xfailed (default OLLAMA_HOST, 943 s). `uvx ruff@0.16.0 format --check .` clean.

## Round 4 (Astra rejected round 3: two new false-negative paths)

### Changes

1. **Word-count core filter removed.** It dropped legitimate paired rows with long text in the lane
   region, which could move the core boundary above a panel heading and hide its deletion. Core
   membership is now: >= 2 numeric words in >= 2 table lanes, and not part of a Notes/Source
   paragraph. The paragraph rule is narrow: a source line whose FIRST word is a Notes/Source opener
   starts a note region; rows in it are not core until a row is again as wide (in lanes) as the table's
   own rows above the opener. Data that resumes after an in-table "Notes:" heading is full-width, so it
   is core again; a paragraph's continuation lines are not. Nothing else (word count, width) changes core
   membership. Regression tests: text-heavy final numeric rows after a panel heading, whose heading
   deletion fires; data resuming after "Notes:" is core (a later dropped label fires); the gomez-shaped
   paragraph (opener line, numeral-bearing continuation lines, dropped line between) does not fire.
2. **Punctuation rule reverted.** `_is_num` is again the verifier's own source predicate plus the
   gate-side leading decimal. `12.5,` is a number on both sides. The sign check compares the leading
   number of the cell and of the source token (`_lead`), so a punctuated `0.230,` still pairs. Tests: a
   table whose values all end in `,` keeps row-order, cell-order and detached-sign checks, and a dropped
   punctuated first or last row fires. (Footnote-star values like `0.253*` are not numeric on either side
   of the verifier, so there is nothing to test there.)
3. **`_PANEL_GAP_ROWS` evidence, reproducible.** New `src/socr/benchmark/ship_gate_gaps.py`, entry point
   `socr-measure-ship-gate-gaps` (pyproject `[project.scripts]`, like the other `socr.benchmark`/measurement
   tools). The span logic moved into `ship_gate.extended_span` so the tool sweeps the same code. Run:
   35 rotated + 92 upright pages, 1815 gaps between consecutive core paired rows.
   - (a) Distribution (gap / block median row gap): median 1.0, p90 2.0, p95 2.25, p99 4.42, max 28.45.
     Pages above p95: panel tables (Fama pp399, 427, 753, 438, 426..., bybee p83 panels A-D, barry p19,
     jiang pp44/50, gow p48 "Panel" headings: panel headings were seen in the earlier page reviews), index
     pages (woodford 786/787/800/802) and others not inspected (tabatabaei 61, sr99 12, herskovic 29,
     gurkaynak 46, perico-ortiz 36, theodoridis 371/545/1203).
   - (b) Missed-panel pages, bound swept 0..10 and unbounded: Fama p398's omitted panels are reached from
     2, brochet p21 (two dropped data rows) from 3, segal p66 (Adj.-R2 and Obs rows) from 5; lopez-lira p32
     and bugel p11 fire at every bound.
   - (c) False extension: pages that fire `data_row_missing` only because of the extension. Bound 2: +Fama
     398. 3: +brochet 21. 5: +segal 66. 6: nothing new. 8 and unbounded: +ljungvist p7 (a table of
     contents: a numeric line that is not a table row, filed as #921). The unbounded structural variant
     ("any full-width row") reaches the same pages as 8, so it does no better on (b) and is worse on (c).
   - Justification: 5 is the smallest bound that reaches every known dropped-row page, 6 adds nothing, and
     the first page a larger bound adds is a false extension. The comment on the constant records this.
     Reproduce: `uv run socr-measure-ship-gate-gaps --rotated-index <index.json> --census <census.jsonl>
     --known 2017__fama__ap.pdf:398 --known lopez_lira_tang_zhu:32 --known bugel_hidalgo:11`.

### Re-measurement

| set | measure | round 3 | round 4 |
|---|---|---|---|
| upright SHIP (92 pages) | pages firing | 20 | 23 |
| | data_row_missing | 4 | 6 |
| | label_row_missing | 17 | 18 |
| | sign_detached / row_order / cell_order | 0 | 0 |
| rotated (35 pages) | wrong pages stopped | 12 / 14 | 12 / 14 (same set; 00, 17 residual) |
| | other pages stopped | 4 / 21 | 6 / 21 (11 and 23 fire again) |
| | label_row_missing wrong / other | 8 / 4 | 9 / 6 |
| | data_row_missing wrong | 2 | 2 |
| | sign_detached wrong | 2 | 2 |

Flips on the upright set, round 3 to round 4:
- Fire again: woodford 787, 791, 802 (index pages) and bybee 67 (a numbered prose list). These are the
  four non-tables of #921; their firing came back with the punctuation rule's removal, and they are out
  of this gate's scope.
- Stopped firing: theodoridis 1203 (not looked at).
- Fable's pages: gomez-cram p10 and piller p33 stay quiet. The four true new7 pages (hack 38, wang 11,
  bugel 11, jiang 44) and all 8 new8 pages still fire. fernandez-fuertes p73 (true) still does not fire
  (Notes below the table).

### Pages that fire (upright SHIP, round 4): basename, page, predicates

| basename | page | predicates |
|---|---|---|
| 2003__woodford.pdf | 787 | data_row_missing |
| 2003__woodford.pdf | 791 | label_row_missing |
| 2003__woodford.pdf | 802 | data_row_missing |
| 2006__boukus_rosenber__information_content_fomc_minutes__WP.pdf | 46 | label_row_missing |
| 2008__faust_wright__efficient_prediction_of_excess_returns.pdf | 44 | label_row_missing |
| 2016__ramey__shocks.pdf | 104 | data_row_missing, label_row_missing |
| 2018__brochet_kolev_lerman__information_transfer_conference_calls__RAS.pdf | 21 | data_row_missing |
| 2020__cieslak_vissing-jorgensen__the_economics_of_fed_put__WP.pdf | 63 | label_row_missing |
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

### Mutation

All killed (canary on `socr.__file__`, uncapped `count == 1`). New or changed guards: sign number compare
exact instead of leading (1), `starts_a_number` (3), rowizer merge condition reverted (2 in the #887 file),
core needs 2 lanes and 2 numerics (1), Notes opener ignored (1), note region never ends (2), word-count rule
re-added (1), trailing punctuation not numeric (2), reach above / below unbounded (1 / 3), reach zero (10),
constant 2 (1), prose bridges the scan (1), plus the unchanged earlier set. The first run left one
survivor (core needs 2 lanes and 2 numerics): the test it relied on had lost its coverage when its lines
started with an opener, and a later patch of mine had deleted it; it is restored without an opener.

### Results

Full suite: 5966 passed, 2 skipped, 4 xfailed (default OLLAMA_HOST, 374 s). `uvx ruff@0.16.0 format --check .` clean.

## Round 5 (Astra rejected round 4: the Notes rule still hides faults)

### Decision: the Notes/Source rule is removed entirely

Astra's three counterexamples to round 4's narrowed rule: (a) a genuine "Source of shock" heading after
four-lane rows, followed by narrower two-lane data, never regained core, so a dropped "Panel B" label or
a dropped numeric row in that section escaped; (b) openers were collected page-wide, so a neighbouring
column's "Notes:" suppressed this table; (c) the resume threshold compared numeric-word counts with
distinct lanes. The rule is deleted. Core membership is: >= 2 numeric words in >= 2 table lanes, nothing
else. **The trade, stated explicitly:** a Notes paragraph swallowed into the grid (gomez-cram p10, a false
DEFER per Fable) now fires again, because a false fire costs one model call and a missed fault can ship a
wrong number. Tests pin the counterexamples as MUST-FIRE: a narrow section after "Source of shock",
"Notes:" or "Source:" (dropped label fires, dropped numeric row fires, intact grid does not); a
neighbouring column's "Notes:" on a row inside the table; and the gomez-shaped paragraph itself, pinned as
an accepted false DEFER. Mutant "Notes rule re-added" fails 5 tests.
Related: the width a row needs to extend the span is now the modal DISTINCT-LANE count of the core rows
(was the modal numeric-word count); a test with two numbers per lane pins it.

### `ship_gate_gaps.py` evidence fixes

- (b) is row level: `--known DOC:PAGE:Y,Y` names the specific omitted source rows and the report says
  whether each lies in a covered span (not whether the page has some missing-row fault).
- (c) lists every row newly covered, and every row newly FIRING, beyond the unextended span, on every page
  including pages that already fire (page, y, row index), plus the increment over the previous bound.
- Bound 0 is a true no-extension baseline: the 10 pt floor and the peer-block gap floor in `extended_span`
  are gone, so the bound is the whole reach (`reach = bound * pitch`). Test: bound 0 extends nothing.
  Before the floors went, lopez-lira p32 and bugel p11 looked reachable at every bound; they were reached
  by the floors, not by the constant.

### Sweep (35 rotated + 92 upright pages; 14 known omitted rows: Fama p398 x4, lopez-lira p32 x2, bugel p11 x4,
brochet p21 x2, segal p66 x2)

| bound (row pitches) | known omitted rows reached | newly FIRING rows beyond baseline (known / other) |
|---|---|---|
| 0 | 0 / 14 | 0 |
| 1 | 0 / 14 | 3 (0 / 3) |
| 2 | 4 / 14 | 7 (4 / 3) |
| 3 | 10 / 14 | 16 (10 / 6) |
| 4 | 12 / 14 | 20 (14 / 6) |
| 5 | **14 / 14** | 26 (16 / 10) |
| 6 | 14 / 14 | 26 (16 / 10) |
| 8 | 14 / 14 | 27 (16 / 11) |
| 10, unbounded | 14 / 14 | 27 (16 / 11) |

"Other" rows at 5: woodford p787/p802 (index pages, a non-table, present from bound 1), ramey p104 (five
rows of a text table that have no grid row), bugel p11 (three more omitted rows of the same missing first
panel, so true). The first row a larger bound adds that is not a real omitted row is ljungvist p7 (a table of
contents) at 8. **5 stays**: it is the smallest bound that reaches every known omitted row (segal p66 needs
5), 6 adds no firing row, and the first false extension is at 8. The unbounded structural variant fires on
exactly the rows 8 does, so it is no better.

### Re-measurement (round 4 to round 5)

| set | measure | round 4 | round 5 |
|---|---|---|---|
| upright SHIP (92) | pages firing | 23 | 25 |
| | data_row_missing | 6 | 6 |
| | label_row_missing | 18 | 20 |
| | sign_detached / row_order / cell_order | 0 | 0 |
| rotated (35) | wrong pages stopped | 12 / 14 | 12 / 14 |
| | other pages stopped | 6 / 21 | 6 / 21 |
| | label_row_missing wrong / other | 9 / 6 | 9 / 6 |
| | data_row_missing wrong / sign_detached wrong | 2 / 2 | 2 / 2 |

Flips, upright: fernandez-fuertes p73 (true per Fable) fires again; gomez-cram p10 (false per Fable) fires,
the accepted trade. Nothing else changes. piller p33 stays quiet. The four true new7 pages and all 8 new8
pages still fire; the rotated 12 of 14 holds with the same set.

### Pages that fire (upright SHIP, round 5): basename, page, predicates

| basename | page | predicates |
|---|---|---|
| 2003__woodford.pdf | 787 | data_row_missing |
| 2003__woodford.pdf | 791 | label_row_missing |
| 2003__woodford.pdf | 802 | data_row_missing |
| 2006__boukus_rosenber__information_content_fomc_minutes__WP.pdf | 46 | label_row_missing |
| 2008__faust_wright__efficient_prediction_of_excess_returns.pdf | 44 | label_row_missing |
| 2016__ramey__shocks.pdf | 104 | data_row_missing, label_row_missing |
| 2018__brochet_kolev_lerman__information_transfer_conference_calls__RAS.pdf | 21 | data_row_missing |
| 2020__cieslak_vissing-jorgensen__the_economics_of_fed_put__WP.pdf | 63 | label_row_missing |
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
| 2025__fernandez-fuertes__monetary_policy_shocks_a_new_hope.pdf | 73 | label_row_missing |
| 2025__gomez-cram_jensen_kung__financial_prediction_markets_a_new_measure_of_earnings_expectations.pdf | 10 | label_row_missing |
| 2025__hack_istrefi_meier__systematic_origins_of_monetary_policy_shocks__WP.pdf | 38 | label_row_missing |
| 2025__wang_liu_chen__current_stance_vs_future_guidance_llm_evidence_on_how_pbc_communication_shapes_the_yield_curve__EL.pdf | 11 | label_row_missing |
| 2026__bugel_hidalgo_luetticke__unconventional_unified_narrative_mp_shocks__WP.pdf | 11 | data_row_missing |
| 2026__jiang_krishnamurthy_lustig_richmond__dollar_erosion_loss_of_reserve_currency_status__WP.pdf | 44 | label_row_missing |
| 2026__jiang_krishnamurthy_lustig_richmond__dollar_erosion_loss_of_reserve_currency_status__WP.pdf | 50 | label_row_missing |

### Mutation

All killed (canary on `socr.__file__`, uncapped `count == 1`), including: Notes rule re-added (5 tests),
word-count rule re-added (2), reach zero (11), constant 2 (1), unbounded reach above / below (1 / 4), prose
bridges the scan (1), core needs 2 lanes and 2 numerics (1), width counted in numeric words instead of lanes
(1, after adding the two-numbers-per-lane test: the first run left it alive), and the unchanged earlier set.

### Results

Full suite: 5972 passed, 2 skipped, 4 xfailed (default OLLAMA_HOST, 1310 s). `uvx ruff@0.16.0 format --check .` clean.

## Round 6 (Astra: removing the peer-block gap floor shrank production coverage)

### Regression and fix

Round 5 dropped the peer-block gap floor so bound 0 would be a true baseline. That also stopped covering a
row omitted from the gap BETWEEN two blocks of one table. Astra's counterexample: two output blocks sharing
four lanes, paired rows at y=100/110/120 and y=300/310/320, an omitted full-width row at y=200; pitch 10,
reach 50, so the row was outside both spans and the gate shipped it.

`table_spans` now keeps two reaches apart:
- OUTWARD, beyond a table's first and last block: governed by `_PANEL_GAP_ROWS` (`extended_span`). Bound 0 means
  no outward extension.
- BETWEEN consecutive blocks of the SAME table: always covered, whatever the bound. The interior has the table
  on both sides. "Same table" is decided from lanes alone (`_same_table_lanes`: every lane of the narrower
  block lies within the snap radius of a lane of the other, and at least `_MIN_LANES_PER_ROW` lanes), never from
  proximity. A row that lies in two blocks' spans is judged once.

Tests: the counterexample must fire `data_row_missing` at bound 0 and at the default bound (and the same page
without the omitted row is clean); the control, a second table whose columns are shifted so its lanes do not
match, with unrelated numeric text between the two tables in the first table's lanes, must not fire.
Mutants: interior coverage removed fails the counterexample; interior applied across non-matching tables fails
the control; lane consistency weakened to one shared lane fails; a row judged twice fails the counterexample.

### Benchmark

`ship_gate_gaps.py` always computes the zero baseline first (the bound list is normalised to start with `0`, and
the comparison refuses to run otherwise), instead of assuming the first custom bound is 0.

### Sweep re-run

Row-level reach of the 14 known omitted rows is unchanged: 0 at bounds 0 and 1, 4 at 2, 10 at 3, 12 at 4,
14 at 5 and above. None of the known rows lies between two blocks, so interior coverage does not reach any at
bound 0. Newly firing rows beyond baseline at 5 fall from 26 to 24 (consistent with two rows now firing at
the bound-0 baseline through interior coverage; I did not identify them), the rest as in round 5: woodford
index rows, ramey p104 rows, three more bugel rows, and the first false extension (ljungvist p7) at 8.
`_PANEL_GAP_ROWS` stays 5, now documented as governing the outward reach only.

### Re-measurement (round 5 to round 6)

| set | measure | round 5 | round 6 |
|---|---|---|---|
| upright SHIP (92) | pages firing | 25 | 25 |
| | data_row_missing / label_row_missing | 6 / 20 | 6 / 20 |
| rotated (35) | wrong / other pages stopped | 12 of 14 / 6 of 21 | 12 of 14 / 6 of 21 |
| | label wrong / other, data wrong, sign wrong | 9 / 6, 2, 2 | 9 / 6, 2, 2 |

No verdict flips on any page set. new7 and new8: unchanged (gomez-cram p10 fires, piller p33 quiet, the four true
new7 pages and all 8 new8 pages fire).

### Pages that fire (upright SHIP, round 6): basename, page, predicates

| basename | page | predicates |
|---|---|---|
| 2003__woodford.pdf | 787 | data_row_missing |
| 2003__woodford.pdf | 791 | label_row_missing |
| 2003__woodford.pdf | 802 | data_row_missing |
| 2006__boukus_rosenber__information_content_fomc_minutes__WP.pdf | 46 | label_row_missing |
| 2008__faust_wright__efficient_prediction_of_excess_returns.pdf | 44 | label_row_missing |
| 2016__ramey__shocks.pdf | 104 | data_row_missing, label_row_missing |
| 2018__brochet_kolev_lerman__information_transfer_conference_calls__RAS.pdf | 21 | data_row_missing |
| 2020__cieslak_vissing-jorgensen__the_economics_of_fed_put__WP.pdf | 63 | label_row_missing |
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
| 2025__fernandez-fuertes__monetary_policy_shocks_a_new_hope.pdf | 73 | label_row_missing |
| 2025__gomez-cram_jensen_kung__financial_prediction_markets_a_new_measure_of_earnings_expectations.pdf | 10 | label_row_missing |
| 2025__hack_istrefi_meier__systematic_origins_of_monetary_policy_shocks__WP.pdf | 38 | label_row_missing |
| 2025__wang_liu_chen__current_stance_vs_future_guidance_llm_evidence_on_how_pbc_communication_shapes_the_yield_curve__EL.pdf | 11 | label_row_missing |
| 2026__bugel_hidalgo_luetticke__unconventional_unified_narrative_mp_shocks__WP.pdf | 11 | data_row_missing |
| 2026__jiang_krishnamurthy_lustig_richmond__dollar_erosion_loss_of_reserve_currency_status__WP.pdf | 44 | label_row_missing |
| 2026__jiang_krishnamurthy_lustig_richmond__dollar_erosion_loss_of_reserve_currency_status__WP.pdf | 50 | label_row_missing |

### Results

All mutants killed (the 25 of round 5 plus 4 new; the one-shared-lane mutant survived the first run and now dies after adding the one-shared-lane control). Full suite: 5976 passed, 2 skipped, 4 xfailed (default OLLAMA_HOST, 2007 s). `uvx ruff@0.16.0 format --check .` clean.

## Round 7 (Astra: two remaining false negatives in the block-interior rule)

1. **Fewer-column panel.** `_same_table_lanes` effectively needed >= 3 lanes (`_MIN_LANES_PER_ROW`) although
   `_table_geometry` accepts a core row with two. A two-lane panel above or below a four-lane one was never
   merged, so an omitted four-lane row between them was outside both spans. The minimum is now the core-row
   minimum (`_MIN_CORE_LANES = 2`, shared by `_table_geometry` and `_same_table_lanes`): a narrower block
   whose lanes ALL align with lanes of the wider block is the same table. Two unrelated tables whose columns
   do not line up still do not merge. MUST-FIRE test: two-lane core rows at y=100/110/120, four-lane at
   300/310/320, an omitted four-lane row at y=200 fires at bound 0 and at the default bound. Controls: the
   non-matching four-lane tables and the one-shared-lane blocks (round 6), and two unrelated two-lane tables
   with shifted lanes.
2. **Dedup blind spot.** A row was marked seen after checking only the first block's lanes, so a three-lane
   block before a matching four-lane block let an interior row that keeps three values and loses the fourth
   pass the narrow check and shielded it from the wider one. Each candidate row is now evaluated ONCE against
   the UNION of the lanes of every block whose span covers it (hits from any applicable lane set, width = the
   widest applicable lane count), then judged. MUST-FIRE test: that shape (the grid keeps the first three
   values as an extra row of the first block); with the fourth value kept too, nothing fires.

Mutants (each reverts one fix): same-table minimum back to `_MIN_LANES_PER_ROW` fails the two-lane/four-lane
test; judging a row against only the first covering block fails the union test; the earlier set (interior
removed, interior across non-matching tables, one-shared-lane) still dies.

### Re-measurement (round 6 to round 7)

| set | measure | round 6 | round 7 |
|---|---|---|---|
| upright SHIP (92) | pages firing | 25 | 25 |
| | data_row_missing / label_row_missing | 6 / 20 | 6 / 20 |
| | sign_detached / row_order / cell_order | 0 | 0 |
| rotated (35) | wrong / other pages stopped | 12 of 14 / 6 of 21 | 12 of 14 / 6 of 21 |
| | label wrong / other, data wrong, sign wrong | 9 / 6, 2, 2 | 9 / 6, 2, 2 |

No verdict flips on any page set; new7 and new8 unchanged (the four true new7 pages and all 8 new8 pages fire,
gomez-cram p10 fires as the accepted trade, piller p33 is quiet). The corpus has no page on which these two
fixes change a verdict; they close synthetic shapes only.

### Pages that fire (upright SHIP, round 7): basename, page, predicates

| basename | page | predicates |
|---|---|---|
| 2003__woodford.pdf | 787 | data_row_missing |
| 2003__woodford.pdf | 791 | label_row_missing |
| 2003__woodford.pdf | 802 | data_row_missing |
| 2006__boukus_rosenber__information_content_fomc_minutes__WP.pdf | 46 | label_row_missing |
| 2008__faust_wright__efficient_prediction_of_excess_returns.pdf | 44 | label_row_missing |
| 2016__ramey__shocks.pdf | 104 | data_row_missing, label_row_missing |
| 2018__brochet_kolev_lerman__information_transfer_conference_calls__RAS.pdf | 21 | data_row_missing |
| 2020__cieslak_vissing-jorgensen__the_economics_of_fed_put__WP.pdf | 63 | label_row_missing |
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
| 2025__fernandez-fuertes__monetary_policy_shocks_a_new_hope.pdf | 73 | label_row_missing |
| 2025__gomez-cram_jensen_kung__financial_prediction_markets_a_new_measure_of_earnings_expectations.pdf | 10 | label_row_missing |
| 2025__hack_istrefi_meier__systematic_origins_of_monetary_policy_shocks__WP.pdf | 38 | label_row_missing |
| 2025__wang_liu_chen__current_stance_vs_future_guidance_llm_evidence_on_how_pbc_communication_shapes_the_yield_curve__EL.pdf | 11 | label_row_missing |
| 2026__bugel_hidalgo_luetticke__unconventional_unified_narrative_mp_shocks__WP.pdf | 11 | data_row_missing |
| 2026__jiang_krishnamurthy_lustig_richmond__dollar_erosion_loss_of_reserve_currency_status__WP.pdf | 44 | label_row_missing |
| 2026__jiang_krishnamurthy_lustig_richmond__dollar_erosion_loss_of_reserve_currency_status__WP.pdf | 50 | label_row_missing |

### Results

Full suite: 5979 passed, 2 skipped, 4 xfailed (default OLLAMA_HOST, 595 s); all 30 mutants killed. `uvx ruff@0.16.0 format --check .` clean.

## Round 7 review outcome: accepted false DEFER (2026-10-01)

Astra flagged an over-merge in 0e52ddd: two unrelated tables whose numeric columns share x positions are linked as one table, so numbers in prose between them read as an omitted interior row and the page DEFERs.

On follow-up, Astra confirmed **NO-FN**: linking only widens spans; pairing, lanes, core rows and outward reach are computed per block before linking; and union-of-lanes can only add requirements. So the over-merge can add false DEFERs but never cause a wrong SHIP.

Under the round-5 policy (a false DEFER costs one model call; a false negative can ship a wrong number) this is accepted. It is pinned as `test_identical_column_separate_tables_are_an_accepted_false_defer`, so any change to it is deliberate.
