# Withhold unverified tables the PDF's own text contradicts (2026-10-03, #1022)

Branch `fix/withhold-contradicted-unverified-tables` from origin/main 416c10a0.

## Why

A strict eye check of 30 ladder-unverified tables (sampled from 75 unverified blocks in 14 documents of
the archive re-OCR) found 12 CORRECT and 18 WRONG. Sign lost: every minus dropped in all 3
Peersman-Smets tables and both Barrot-Sauvagnat tables, so the numbers shipped inverted. Others: row
shift 5, missing or detached header 4, wrong number 3, header misbound 3, phantom table 2, shredded
caption 2. "Unverified" meant a table that ships with a flag; the owner approved withholding the ones
the PDF's own words contradict. The corpus is copyrighted: this log reports pages and counts only.
Evidence: `~/.local/state/socr-housekeeping/jev-eval/{RESULT.md,verdicts.jsonl,sample.json}`; phase-1
numbers in `.../jev-eval/WITHHOLD-MEASURE.md`.

## Phase 1: measurement (no src change)

Three checks over the page's native text layer, each abstaining (`no_evidence`) where the layer is
unusable. Re-run on the production module after it was written, so these are the shipped numbers.

| check | WRONG flagged | CORRECT flagged (of 12) | WRONG no_evidence | CORRECT no_evidence | WRONG clear | CORRECT clear |
|---|---|---|---|---|---|---|
| sign | 5 (all 5 sign-lost) | 0 | 1 | 9 | 12 | 3 |
| row_shift | 1 | 0 | 11 | 10 | 6 | 2 |
| number_absent | 1 | 0 | 10 | 12 | 7 | 0 |
| any | 6 of 18 | 0 | | | | |

- Per defect class: sign-lost 5/5, row shift 1/5, wrong number 1/3, everything else 0 (not attempted).
- All 75 blocks: sign 15 contradicted / 31 clear / 29 no evidence; row_shift 2 / 21 / 52; number_absent
  1 / 20 / 54. Union flagged 17 of 75; no evidence on all three 28 of 75.
- The 11 flagged blocks outside the sample are Peersman p11, p20; Barrot-Sauvagnat p22, 23, 27, 35, 37,
  38, p40 block 2; Gofman p38; Bernanke p23. Gofman p38 and Bernanke p23 were checked by eye on the
  render and are real (an invented minus on the I/K Data CAPM-beta cell; "Other" split from its values
  onto an unlabeled row). Peersman and Barrot are the documents where every sampled table lost its signs.
- Stop criteria (a check flagging more than 1 of the 12 CORRECT, or sign catching fewer than 5): not hit.
- Weak side of the measurement, stated plainly: only 3 of the 12 CORRECT tables carry usable text-layer
  evidence (8 sit on invisible OCR layers under a raster, Forsythe/Hansen/Gleason, and Gofman 45/46
  print numbers the layer lacks). 0/12 is therefore a weak false-positive bound by itself; the off-sample
  flags above are the stronger check, and the two verified by eye were both true.
- A prototype bug found and fixed on the way: the printed-word parser kept the minus sign while the
  table-side parser did not, so a row with any negative cell never matched its printed line. The
  prototype's row-shift flag on sampled id 11 was an artifact of that; with both sides unsigned the
  flag set is id 14 and Bernanke p23. The unit tests use rows containing negatives for this reason.

## Phase 2: what changed

- `src/socr/tables/native_contradiction.py` (new): the three checks and `contradictions_for_tables`.
  Evidence rule: abstain when most of the page's characters are invisible (render mode 3) or when more
  than `EXTRA_NUMBERS_MAX_SHARE` (row_corroboration's existing 0.02) of the table's numbers are absent
  from the layer. Sign is read at character level because word extraction loses the unmapped-glyph (#990)
  and minus-as-"2" (#913) forms; it also reads the `- 0.48` form (#930). Row-shift and number-absent need
  the table's region from `locate_tables` and only run when boxes == blocks; row-shift also needs upright
  text (`ship_gate._SAME_TEXT_DIRECTION_TOL_RAD`). No new threshold: the tolerances are existing named
  quantities; the only cut introduced is a majority (what "mostly invisible" means).
- `orchestrator._withhold_contradicted_unverified_tables`, called right after `_run_table_judge_gate`
  (before the existing page-PNG render for WITHHELD pages, so the marker gets its image). It targets the
  tables whose latest ladder terminal is UNVERIFIED or absent (absent = backfilled UNVERIFIED at
  assemble). ACCEPTED tables are never touched; a page already REJECTED/WITHHELD keeps that verdict; a
  failure of the check keeps today's behaviour. On a finding it sets the page disposition to
  `TABLE_WITHHELD` and appends a `table_ladder_withheld` event with `data["reason"] =
  "native_contradiction"` and `data["contradictions"]` naming each kind and finding.
- Everything downstream already handles `TABLE_WITHHELD`: `manifest._apply_ladder_disposition_guard`
  ships the regional floor (marker plus page image, prose kept under GH-520's coverage proof, else the
  whole page floors), the page fails ERROR/`table_withheld`, the document goes AUDIT_FAILED/ERROR through
  `table_withheld_pages`, and `table_counts` counts the marker as withheld and the block as neither shipped
  nor unverified.
- Wording: the metadata note (`_table_judge_ladder_note`) and the CLI line previously described every
  withhold as "the readers rejected it and a blind cell transcription read different tokens". That is false
  for this one, so both split on `_native_contradiction_withheld_pages` (read from the events, which survive
  resume) and say "the ladder could not verify the table and the PDF's own text contradicts it" with the
  kinds. `REASON_NATIVE_CONTRADICTION` lives in `judge/table_verdict.py`.

## Behaviour changes worth a reviewer's eye

- Withholding is page-granular, as it already is for a ladder withhold: every table region on the page
  ships as the marker, including a sibling table the ladder ACCEPTED. A per-table splice would need the
  floor to take a table subset; not done here.
- `tests/fixtures/table_ladder` page 2 is a genuine row-label shift. With the new check live it ends
  WITHHELD instead of UNVERIFIED. `test_ladder_e2e.py` pins the ladder's own terminals and wording, so its
  `_run` helper now stubs the new check by default and one new class
  (`TestContradictedShiftIsWithheld`) pins the delta on that same fixture, the check being the only
  difference.
- The ladder-off path is unchanged: the hook is inside the `table_judge_ladder` block.
- Resume: the withheld page is restored as a content terminal like any ladder withhold (disposition from
  the sidecar, events replayed because `table_ladder_withheld` is already in `TABLE_LADDER_EVENT_KINDS`).

## Tests

`tests/test_native_contradiction.py` (hermetic: synthetic PDFs drawn in the test; the pipeline tests patch
`_available_engines_for_agentic`, `_resolve_judge_model` to "" and the rungs; no ollama, no provider).
Differences, not absolutes:

- per check, the same table with and without the fault: dropped minus, invented minus, permuted label
  column, values on an unlabelled row, one wrong cell; a sibling table's copy of a value is not this
  table's fault;
- abstention: invisible OCR layer, a table the layer does not carry, an extraction failure;
- the character-level forms on a stand-in page (U+2212, en dash, `- 0.48`, control byte, minus-as-"2",
  `1990-2000`, `SIEM50`);
- `process()` on the same page, table text differing only by the fault, both ending the ladder UNVERIFIED
  (no rung): page disposition, the shipped page text, the event's kind, the #993 counts (unverified 1 /
  withheld 0 becomes unverified 0 / withheld 1 / shipped 0), the document status and error, `metadata.json`
  and the CLI summary, and the row-shift and wrong-number variants;
- scope: ACCEPTED untouched, the same page UNVERIFIED withheld, REJECTED keeps its stronger verdict,
  ladder off leaves the table alone.

Mutation (out-of-repo copy of src + tests + pyproject, a canary test asserting `socr.__file__` is inside
the copy and passing after each mutation, uncapped `anchor.count == 1` before editing, suite =
`test_native_contradiction.py` + `test_ladder_e2e.py`). All eight killed:

| mutant | killed by |
|---|---|
| sign: dropped-minus branch `= 0` | `TestSign::test_dropped_minus_is_the_only_difference` |
| sign: invented-minus branch `= 0` | `TestSign::test_invented_minus_is_the_reverse` |
| row_shift: other-row binding test `if False` | `TestRowShift::test_a_permuted_label_column_is_the_only_difference` |
| number_absent: always clear | `TestNumberAbsent::test_one_wrong_cell_is_the_only_difference` |
| evidence gate: invisible treated as visible | `TestAbstention::test_an_invisible_ocr_layer_proves_nothing` |
| wiring: hook not called | `TestWithholdThroughTheWholePipeline::test_a_dropped_minus_...` |
| scope: accepted tables not exempt | `TestScope::test_an_accepted_table_is_never_withheld` |
| surface: contradiction reason not read | `TestWithholdThroughTheWholePipeline::test_a_dropped_minus_...` |

## Not covered, on purpose

Header misbound/missing, shredded captions, phantom tables and row-count errors: the existing helpers
cannot test them without flagging tables a reader would call correct (the ship_gate predicates for them
fire on 1 to 4 of the 12 CORRECT tables each). Row-shift recall is low (1 of 5 sampled): the two
Bernanke tables have right values on the right lines with a detached header/rows, which a label-binding
rule does not see. The check does not run on scanned or invisible-text PDFs; a text-less rotated scan keeps
today's UNVERIFIED flag.

## Results

Full suite: 6824 passed, 2 skipped, 4 xfailed (268 s). `uvx ruff@0.16.0 format --check .` clean (869 files).
The first full run had 5 failures, all the same cause and all expected: four process-level tests in
`test_gh367_adjudication_lift.py`, `test_gh974_review_pins.py` and `test_ladder_binding_evidence.py` pin the
ladder's binding clamp on the row-shifted fixture page, which the new check now withholds. Their pipeline
setups stub `_withhold_contradicted_unverified_tables` (commented at each site); the withhold on that
fixture is pinned once, with the check as the only difference, in `test_ladder_e2e.py::TestContradictedShiftIsWithheld`.

## Round 2 (Astra rejected ed6131cd)

1. **Sign false positives.** One scanner, `scan_numbers`, now reads sign and value from page text and from table
   cells alike. A bracketed number is its own class (`paren`): it matches either sign on the other side, so
   `(0.12)` equals `-0.12`, and a table that prints `4.98` where the page prints a t-statistic `(4.98)` is not
   convicted either. A dash is a range separator, not a minus, when it follows a digit (or `)`/`]`/`%`) directly
   (`1-2`, `1–2`) or after a space with a space after it (`1 – 2`); `0.45 -0.07` and `Mean – 0.48` stay negatives.
2. **Footnotes and duplicates.** Footnote marks (stars, dagger, section sign, superscript digits) are stripped from
   the edges of a printed word before it is read as a number, so a marked value no longer vanishes from the printed
   line. Row-shift abstains for any row whose value multiset appears in more than one row of the grid.
3. **Wrong number.** Chose to normalise locally, not to change `row_corroboration` (its `1,234` vs `1234` token
   comparison feeds the row-corroboration gate and other consumers). `number_absent_contradictions` now compares
   unsigned values through `scan_numbers` against the region's page numbers read at character level, so thousands
   separators fold, footnote marks are ignored, and a minus drawn as a "2" (#913) is not read as the number 2.x.
4. **Siblings reported.** Every table the page-granular floor removes now has a `table_ladder_withheld` record: the
   contradicted one with `reason=native_contradiction`, each other with `reason=sibling_of_contradicted`,
   `contradicted_tables` and `prior_terminal` (accepted / unverified / none). `table_counts.count_page_tables` takes
   the number of distinct tables the page's withheld events name (live events and sidecar `audit_events` through
   `withheld_table_events`) and counts `max(markers, events)`, so a whole-page floor with two removed tables counts
   two, not one.
5. **Cost of page-granular withholding on the 75 blocks: 0.** 18 of 75 blocks are flagged, on 16 pages; every other
   table on those pages is itself flagged. No accepted, verified or merely unflagged sibling table is removed. (The
   sample has no page where the question arises, so this says nothing about a corpus with more tables per page.)

Phase 1 re-run on the final module. 30 sampled: sign 5/5 sign-lost, row shift 2 of 5 (ids 11 and 14; id 11 returns now
that starred values are read), wrong number 1, **0 of 12 CORRECT flagged by any check**. All 75: sign 15 / row_shift 3 /
number_absent 1 contradicted; union 18; no evidence on all three 28.

Mutants killed (same harness): bracketed read as positive, range read as minus, footnote marks not stripped, duplicate
rows convict, thousands separator not folded, sibling gets no record, metric ignores per-table events.
